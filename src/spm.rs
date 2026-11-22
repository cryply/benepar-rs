// SentencePiece model (spiece.model) to a `tokenizers` Unigram tokenizer.
//
// The .model file is a protobuf ModelProto. Only the fields needed here are
// read: the pieces (text, score, type) and the precompiled normalization map.
// The tokenizer is assembled the same way the HuggingFace converter builds
// T5TokenizerFast: precompiled normalizer, collapse runs of spaces, Metaspace
// pre-tokenizer, Unigram model.

use anyhow::{bail, Context, Result};
use tokenizers::models::unigram::Unigram;
use tokenizers::normalizers::{NormalizerWrapper, Precompiled, Replace, Sequence};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::Tokenizer;

const PIECE_UNKNOWN: u64 = 2;

pub struct SpmModel {
    pub pieces: Vec<(String, f64, u64)>,
    pub charsmap: Vec<u8>,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64> {
        let mut out = 0u64;
        let mut shift = 0;
        loop {
            let b = *self.buf.get(self.pos).context("truncated varint")?;
            self.pos += 1;
            out |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(out);
            }
            shift += 7;
            if shift > 63 {
                bail!("varint too long");
            }
        }
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("length overflow")?;
        let s = self.buf.get(self.pos..end).context("truncated field")?;
        self.pos = end;
        Ok(s)
    }

    // Returns (field number, wire type, payload). Varints come back as their
    // little-endian bytes so the caller can decode by wire type.
    fn field(&mut self) -> Result<(u64, u64, Field<'a>)> {
        let key = self.varint()?;
        let (num, wt) = (key >> 3, key & 7);
        let f = match wt {
            0 => Field::Varint(self.varint()?),
            1 => Field::Fixed(self.bytes(8)?),
            2 => {
                let n = self.varint()? as usize;
                Field::Bytes(self.bytes(n)?)
            }
            5 => Field::Fixed(self.bytes(4)?),
            _ => bail!("unsupported wire type {wt}"),
        };
        Ok((num, wt, f))
    }
}

enum Field<'a> {
    Varint(u64),
    Fixed(&'a [u8]),
    Bytes(&'a [u8]),
}

impl SpmModel {
    pub fn parse(buf: &[u8]) -> Result<Self> {
        let mut pieces = Vec::new();
        let mut charsmap = Vec::new();
        let mut r = Reader::new(buf);
        while !r.done() {
            match r.field()? {
                (1, _, Field::Bytes(b)) => pieces.push(parse_piece(b)?),
                (3, _, Field::Bytes(b)) => {
                    let mut n = Reader::new(b);
                    while !n.done() {
                        if let (2, _, Field::Bytes(m)) = n.field()? {
                            charsmap = m.to_vec();
                        }
                    }
                }
                _ => {}
            }
        }
        if pieces.is_empty() {
            bail!("no pieces in sentencepiece model");
        }
        Ok(Self { pieces, charsmap })
    }

    pub fn tokenizer(&self) -> Result<Tokenizer> {
        let unk_id = self
            .pieces
            .iter()
            .position(|p| p.2 == PIECE_UNKNOWN)
            .context("sentencepiece model has no <unk> piece")?;
        let vocab = self
            .pieces
            .iter()
            .map(|(s, score, _)| (s.clone(), *score))
            .collect();
        let model = Unigram::from(vocab, Some(unk_id), false)
            .map_err(|e| anyhow::anyhow!("unigram: {e}"))?;
        let mut tk = Tokenizer::new(model);

        let mut norms: Vec<NormalizerWrapper> = Vec::new();
        if !self.charsmap.is_empty() {
            let pc = Precompiled::from(&self.charsmap)
                .map_err(|e| anyhow::anyhow!("precompiled charsmap: {e}"))?;
            norms.push(pc.into());
        }
        let collapse = Replace::new(
            tokenizers::normalizers::replace::ReplacePattern::Regex(" {2,}".into()),
            " ",
        )
        .map_err(|e| anyhow::anyhow!("replace: {e}"))?;
        norms.push(collapse.into());
        tk.with_normalizer(Some(Sequence::new(norms)))
            .map_err(|e| anyhow::anyhow!("normalizer: {e}"))?;
        tk.with_pre_tokenizer(Some(Metaspace::new('\u{2581}', PrependScheme::Always, true)));
        Ok(tk)
    }

    pub fn id(&self, piece: &str) -> Option<u32> {
        self.pieces.iter().position(|p| p.0 == piece).map(|i| i as u32)
    }
}

fn parse_piece(b: &[u8]) -> Result<(String, f64, u64)> {
    let mut r = Reader::new(b);
    let (mut text, mut score, mut kind) = (String::new(), 0f64, 1u64);
    while !r.done() {
        match r.field()? {
            (1, _, Field::Bytes(s)) => text = String::from_utf8(s.to_vec())?,
            (2, 5, Field::Fixed(f)) => score = f32::from_le_bytes(f.try_into()?) as f64,
            (3, _, Field::Varint(v)) => kind = v,
            _ => {}
        }
    }
    Ok((text, score, kind))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip() {
        let buf = [0xac, 0x02];
        let mut r = Reader::new(&buf);
        assert_eq!(r.varint().unwrap(), 300);
    }
}
