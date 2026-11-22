// The benepar_en3 network: T5-small features, a 2-layer partitioned
// transformer (content and position halves kept apart), and MLP heads that
// score every span label and every POS tag.
//
// Shapes follow the Python model for a batch of one sentence of W words.
// Positions 0 and W+1 are the start and stop markers taken from T5.

use crate::spm::SpmModel;
use crate::t5;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{layer_norm, linear, linear_no_bias, LayerNorm, Linear, VarBuilder};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use tokenizers::Tokenizer;

const LN_EPS: f64 = 1e-5;
const T5_PAD: u32 = 0;
const T5_EOS: u32 = 1;

#[derive(Clone)]
struct Mlp {
    l0: Linear,
    norm: LayerNorm,
    l1: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, d_in: usize, d_hidden: usize, d_out: usize) -> Result<Self> {
        Ok(Self {
            l0: linear(d_in, d_hidden, vb.pp("0"))?,
            norm: layer_norm(d_hidden, LN_EPS, vb.pp("1"))?,
            l1: linear(d_hidden, d_out, vb.pp("3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm.forward(&self.l0.forward(x)?)?.relu()?;
        Ok(self.l1.forward(&h)?)
    }
}

#[derive(Clone)]
struct Layer {
    w_qkv_c: Tensor,
    w_qkv_p: Tensor,
    w_o_c: Tensor,
    w_o_p: Tensor,
    ff1_c: Linear,
    ff1_p: Linear,
    ff2_c: Linear,
    ff2_p: Linear,
    norm_attn: LayerNorm,
    norm_ff: LayerNorm,
    n_head: usize,
    d_qkv: usize,
}

impl Layer {
    fn load(vb: VarBuilder, d_model: usize, n_head: usize, d_qkv: usize, d_ff: usize) -> Result<Self> {
        let half = d_model / 2;
        let a = vb.pp("self_attn");
        Ok(Self {
            w_qkv_c: a.get((n_head, half, 3, d_qkv / 2), "w_qkv_c")?,
            w_qkv_p: a.get((n_head, half, 3, d_qkv / 2), "w_qkv_p")?,
            w_o_c: a.get((n_head, d_qkv / 2, half), "w_o_c")?,
            w_o_p: a.get((n_head, d_qkv / 2, half), "w_o_p")?,
            ff1_c: linear(half, d_ff / 2, vb.pp("linear1.linear_c"))?,
            ff1_p: linear(half, d_ff / 2, vb.pp("linear1.linear_p"))?,
            ff2_c: linear(d_ff / 2, half, vb.pp("linear2.linear_c"))?,
            ff2_p: linear(d_ff / 2, half, vb.pp("linear2.linear_p"))?,
            norm_attn: layer_norm(d_model, LN_EPS, vb.pp("norm_attn"))?,
            norm_ff: layer_norm(d_model, LN_EPS, vb.pp("norm_ff"))?,
            n_head,
            d_qkv,
        })
    }

    // x_part: (L, half). Returns q, k, v for one partition, each (H, L, d_qkv/2).
    fn qkv(&self, x_part: &Tensor, w: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (h, half, a) = (self.n_head, x_part.dim(1)?, self.d_qkv / 2);
        let l = x_part.dim(0)?;
        let w = w.reshape((h, half, 3 * a))?;
        let x = x_part.unsqueeze(0)?.broadcast_as((h, l, half))?.contiguous()?;
        let y = x.matmul(&w)?.reshape((h, l, 3, a))?;
        let q = y.i((.., .., 0, ..))?.contiguous()?;
        let k = y.i((.., .., 1, ..))?.contiguous()?;
        let v = y.i((.., .., 2, ..))?.contiguous()?;
        Ok((q, k, v))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let half = x.dim(1)? / 2;
        let x_c = x.narrow(1, 0, half)?;
        let x_p = x.narrow(1, half, half)?;
        let (q_c, k_c, v_c) = self.qkv(&x_c, &self.w_qkv_c)?;
        let (q_p, k_p, v_p) = self.qkv(&x_p, &self.w_qkv_p)?;
        let scale = 1.0 / (self.d_qkv as f64).sqrt();
        let q = (Tensor::cat(&[&q_c, &q_p], D::Minus1)? * scale)?;
        let k = Tensor::cat(&[&k_c, &k_p], D::Minus1)?;
        let v = Tensor::cat(&[&v_c, &v_p], D::Minus1)?;
        let dots = q.matmul(&k.t()?.contiguous()?)?;
        let probs = candle_nn::ops::softmax_last_dim(&dots)?;
        let o = probs.matmul(&v)?;
        let a = self.d_qkv / 2;
        let o_c = o.narrow(D::Minus1, 0, a)?.contiguous()?;
        let o_p = o.narrow(D::Minus1, a, a)?.contiguous()?;
        let out_c = o_c.matmul(&self.w_o_c)?.sum(0)?;
        let out_p = o_p.matmul(&self.w_o_p)?.sum(0)?;
        let residual = Tensor::cat(&[&out_c, &out_p], 1)?;
        let x = self.norm_attn.forward(&(x + residual)?)?;

        let x_c = x.narrow(1, 0, half)?;
        let x_p = x.narrow(1, half, half)?;
        let h_c = self.ff1_c.forward(&x_c)?.relu()?;
        let h_p = self.ff1_p.forward(&x_p)?.relu()?;
        let r_c = self.ff2_c.forward(&h_c)?;
        let r_p = self.ff2_p.forward(&h_p)?;
        let residual = Tensor::cat(&[&r_c, &r_p], 1)?;
        Ok(self.norm_ff.forward(&(x + residual)?)?)
    }
}

pub struct Output {
    // scores[i][j][label] for the span of words i..=j (j >= i).
    pub scores: Vec<Vec<Vec<f32>>>,
    pub tags: Vec<String>,
}

/// The loaded model. Cloning is cheap: the weights are shared.
#[derive(Clone)]
pub struct Parser {
    tokenizer: Tokenizer,
    t5: t5::T5ForConditionalGeneration,
    project: Linear,
    timing: Tensor,
    timing_norm: LayerNorm,
    layers: Vec<Layer>,
    f_label: Mlp,
    f_tag: Mlp,
    d_model: usize,
    pub labels: Vec<String>,
    pub tags: Vec<String>,
    pub force_root: bool,
    device: Device,
}

fn vocab(v: &Value) -> Result<Vec<String>> {
    let m = v.as_object().context("vocab is not an object")?;
    let mut by_id = BTreeMap::new();
    for (k, id) in m {
        by_id.insert(id.as_u64().context("vocab id")? as usize, k.clone());
    }
    let n = by_id.keys().last().map(|x| x + 1).unwrap_or(0);
    let mut out = vec![String::new(); n];
    for (i, k) in by_id {
        out[i] = k;
    }
    Ok(out)
}

fn hp(h: &Value, key: &str) -> Result<usize> {
    h[key].as_u64().map(|x| x as usize).with_context(|| format!("hparam {key}"))
}

impl Parser {
    // dir must contain config.json, spiece.model and benepar_model.bin
    // (the PyTorch file from the benepar_en3 release, decompressed).
    pub fn load(dir: &Path) -> Result<Self> {
        let device = Device::Cpu;
        let cfg_text = std::fs::read_to_string(dir.join("config.json"))?;
        let cfg: Value = serde_json::from_str(&cfg_text)?;
        let b = &cfg["benepar"];
        let h = &b["hparams"];
        // benepar_en3 uses t5-small, benepar_en3_large t5-large; all sizes
        // below come from config.json.
        let pretrained = h["pretrained_model"].as_str().unwrap_or("");
        if !pretrained.starts_with("t5-") || h["use_encoder"] != true {
            bail!("only the benepar_en3 layouts (T5 + encoder) are supported, got {pretrained:?}");
        }
        let d_model = hp(h, "d_model")?;
        let (n_head, d_kv, d_ff) = (hp(h, "num_heads")?, hp(h, "d_kv")?, hp(h, "d_ff")?);
        let n_layers = hp(h, "num_layers")?;
        let max_len = hp(h, "encoder_max_len")?;
        let labels = vocab(&b["label_vocab"])?;
        let tags = vocab(&b["tag_vocab"])?;
        let force_root = h["force_root_constituent"].as_bool().unwrap_or(true);

        let t5_cfg: t5::Config = serde_json::from_str(&cfg_text)?;
        let spm = SpmModel::parse(&std::fs::read(dir.join("spiece.model"))?)?;
        let tokenizer = spm.tokenizer()?;

        let weights = dir.join("benepar_model.bin");
        let vb = VarBuilder::from_pth(&weights, DType::F32, &device)
            .with_context(|| format!("loading {}", weights.display()))?;
        let t5 = t5::T5ForConditionalGeneration::load(vb.pp("pretrained_model"), &t5_cfg)?;
        let project = linear_no_bias(t5_cfg.d_model, d_model / 2, vb.pp("project_pretrained"))?;
        let timing = vb.get((max_len, d_model / 2), "add_timing.timing_table")?;
        let timing_norm = layer_norm(d_model, LN_EPS, vb.pp("add_timing.norm"))?;
        let layers = (0..n_layers)
            .map(|i| Layer::load(vb.pp(format!("encoder.layers.{i}")), d_model, n_head, d_kv, d_ff))
            .collect::<Result<Vec<_>>>()?;
        let f_label = Mlp::load(vb.pp("f_label"), d_model, hp(h, "d_label_hidden")?, labels.len() - 1)?;
        let f_tag = Mlp::load(vb.pp("f_tag"), d_model, hp(h, "d_tag_hidden")?, tags.len())?;
        Ok(Self {
            tokenizer,
            t5,
            project,
            timing,
            timing_norm,
            layers,
            f_label,
            f_tag,
            d_model,
            labels,
            tags,
            force_root,
            device,
        })
    }

    // Subword ids for the sentence and, per word, the index of the last
    // subword overlapping it (same rule as benepar's retokenize()).
    fn retokenize(&self, words: &[String], space_after: &[bool]) -> Result<(Vec<u32>, Vec<usize>)> {
        let mut s = String::new();
        let mut spans = Vec::with_capacity(words.len());
        for (w, sp) in words.iter().zip(space_after) {
            let start = s.chars().count();
            s.push_str(w);
            spans.push((start, start + w.chars().count()));
            if *sp {
                s.push(' ');
            }
        }
        let enc = self
            .tokenizer
            .encode_char_offsets(s.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        let toks: Vec<(usize, (usize, usize))> = enc
            .get_offsets()
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .collect();
        if toks.is_empty() {
            bail!("empty sentence");
        }
        let mut wft = vec![0usize; words.len()];
        let mut t = 0usize;
        for (wi, &(ws, we)) in spans.iter().enumerate() {
            while t < toks.len() && toks[t].1 .1 <= ws {
                t += 1;
            }
            if t >= toks.len() {
                bail!("tokenizer offsets do not cover word {wi}");
            }
            if toks[t].1 .1 > we {
                wft[wi] = toks[t].0;
            }
            while t < toks.len() && toks[t].1 .1 <= we {
                wft[wi] = toks[t].0;
                t += 1;
            }
        }
        ids.push(T5_EOS);
        Ok((ids, wft))
    }

    pub fn parse(&mut self, words: &[String], space_after: &[bool]) -> Result<Output> {
        let n_words = words.len();
        let (ids, wft) = self.retokenize(words, space_after)?;
        if ids.len() + 1 > 512 || n_words + 2 > self.timing.dim(0)? {
            bail!("sentence too long ({n_words} words, {} subwords)", ids.len());
        }

        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut dec_ids = vec![T5_PAD];
        dec_ids.extend_from_slice(&ids);
        let dec_input = Tensor::new(dec_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        self.t5.clear_kv_cache();
        let enc_out = self.t5.encode(&input)?;
        let hidden = self.t5.decode_hidden(&dec_input, &enc_out)?.squeeze(0)?;
        self.t5.clear_kv_cache();

        // Decoder position p sees the T5 token p-1, hence the +1 shift; the
        // first and last positions stand for the start and stop markers.
        let mut pick: Vec<u32> = Vec::with_capacity(n_words + 2);
        pick.push(0);
        pick.extend(wft.iter().map(|&t| (t + 1) as u32));
        pick.push(ids.len() as u32);
        let pick = Tensor::new(pick.as_slice(), &self.device)?;
        let feats = hidden.index_select(&pick, 0)?;

        let len = n_words + 2;
        let content = self.project.forward(&feats)?;
        let timing = self.timing.narrow(0, 0, len)?;
        let mut x = self.timing_norm.forward(&Tensor::cat(&[&content, &timing], 1)?)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // Even features first, then odd ones, as in the Python model.
        let order: Vec<u32> = (0..self.d_model as u32)
            .step_by(2)
            .chain((1..self.d_model as u32).step_by(2))
            .collect();
        let ann = x.index_select(&Tensor::new(order.as_slice(), &self.device)?, 1)?;

        let tag_ids = self.f_tag.forward(&ann)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
        let tags = tag_ids[1..=n_words]
            .iter()
            .map(|&i| self.tags.get(i as usize).cloned().unwrap_or_else(|| "UNK".into()))
            .collect();

        let half = self.d_model / 2;
        let fence = Tensor::cat(
            &[
                &ann.narrow(0, 0, len - 1)?.narrow(1, 0, half)?,
                &ann.narrow(0, 1, len - 1)?.narrow(1, half, half)?,
            ],
            1,
        )?;
        let ends = fence.narrow(0, 1, n_words)?.unsqueeze(0)?;
        let starts = fence.narrow(0, 0, n_words)?.unsqueeze(1)?;
        let span = ends.broadcast_sub(&starts)?.reshape((n_words * n_words, self.d_model))?;
        let logits = self.f_label.forward(&span)?;
        let zeros = Tensor::zeros((n_words * n_words, 1), DType::F32, &self.device)?;
        let logits = Tensor::cat(&[&zeros, &logits], 1)?;
        let flat = logits.to_vec2::<f32>()?;

        let mut scores = vec![Vec::new(); n_words];
        for (i, row) in scores.iter_mut().enumerate() {
            *row = (0..n_words).map(|j| flat[i * n_words + j].clone()).collect();
        }
        Ok(Output { scores, tags })
    }
}
