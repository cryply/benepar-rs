mod chart;
mod model;
#[allow(dead_code)]
mod spm;
#[allow(dead_code)]
mod t5;
mod text;
#[allow(dead_code)]
mod tree;

use anyhow::{Context, Result};
use clap::Parser as _;
use std::io::Read;
use std::path::PathBuf;

/// Constituency parser: benepar_en3 inference in Rust.
#[derive(clap::Parser)]
struct Args {
    /// Directory with config.json, spiece.model and benepar_model.bin
    #[arg(long, env = "BENEPAR_MODEL")]
    model: PathBuf,
    /// Number of parse trees per sentence (best first)
    #[arg(short, long, default_value_t = 1)]
    k: usize,
    /// Print the score of every tree
    #[arg(long)]
    scores: bool,
    /// Labels tried per span when -k > 1; 1 means trees differ only in
    /// bracketing (attachment), more also varies the labels
    #[arg(long, default_value_t = 1)]
    label_variants: usize,
    /// Text to parse; read from stdin when missing
    text: Option<String>,
}

fn prepare_weights(dir: &std::path::Path) -> Result<()> {
    let bin = dir.join("benepar_model.bin");
    let bz2 = dir.join("benepar_model.bin.bz2");
    if bin.exists() || !bz2.exists() {
        return Ok(());
    }
    eprintln!("decompressing {} ...", bz2.display());
    let mut src = bzip2::read::BzDecoder::new(std::fs::File::open(&bz2)?);
    let tmp = dir.join("benepar_model.bin.part");
    let mut dst = std::fs::File::create(&tmp)?;
    std::io::copy(&mut src, &mut dst)?;
    std::fs::rename(&tmp, &bin)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    prepare_weights(&args.model)?;
    let text = match args.text {
        Some(t) => t,
        None => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        }
    };
    let mut parser = model::Parser::load(&args.model).context("loading model")?;
    for sent in text::sentences(&text) {
        let out = parser.parse(&sent.words, &sent.space_after)?;
        let leaves: Vec<tree::Node> = sent
            .words
            .iter()
            .zip(&out.tags)
            .map(|(w, t)| tree::Node::leaf(t, w))
            .collect();
        // Ask for more derivations than needed: several binarizations of
        // the same n-ary tree score the same and print the same.
        let c = chart::Chart::decode(&out.scores, args.k * 4, args.label_variants, parser.force_root);
        let mut seen: Vec<String> = Vec::new();
        for r in 0..c.len() {
            let t = c.tree(r, &parser.labels, &leaves).to_string();
            if seen.contains(&t) {
                continue;
            }
            if args.scores {
                println!("{:.3}\t{t}", c.score(r));
            } else {
                println!("{t}");
            }
            seen.push(t);
            if seen.len() == args.k {
                break;
            }
        }
        if args.k > 1 {
            println!();
        }
    }
    Ok(())
}
