mod chart;
mod model;
#[allow(dead_code)]
mod spm;
#[allow(dead_code)]
mod t5;
mod text;
mod treebank;
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
    /// Score the parser on a treebank file (one bracketed tree per line)
    /// and print EVALB-style precision, recall and F1 instead of parsing text
    #[arg(long, value_name = "FILE")]
    eval: Option<PathBuf>,
    /// With --eval: stop after this many sentences
    #[arg(long)]
    limit: Option<usize>,
    /// Text to parse; read from stdin when missing
    text: Option<String>,
}

fn evaluate(parser: &mut model::Parser, file: &std::path::Path, limit: Option<usize>) -> Result<()> {
    let data = std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
    let mut counts = treebank::Counts::default();
    let mut skipped = 0usize;
    let started = std::time::Instant::now();
    for (n, line) in data.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        if limit.is_some_and(|l| n >= l) {
            break;
        }
        let gold = treebank::parse(line).with_context(|| format!("tree on line {}", n + 1))?;
        let words: Vec<String> = treebank::leaves(&gold)
            .iter()
            .map(|(_, w)| treebank::unescape(w))
            .collect();
        let spaces = treebank::guess_spaces(&words);
        let out = match parser.parse(&words, &spaces) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("line {}: {e}", n + 1);
                skipped += 1;
                continue;
            }
        };
        let leaves: Vec<tree::Node> = words
            .iter()
            .zip(&out.tags)
            .map(|(w, t)| tree::Node::leaf(t, w))
            .collect();
        let c = chart::Chart::decode(&out.scores, 1, 1, parser.force_root);
        let pred = c.tree(0, &parser.labels, &leaves);
        counts.add(&gold, &pred);
        if (n + 1) % 200 == 0 {
            eprintln!("{} sentences, F1 {:.2}", n + 1, counts.f1());
        }
    }
    let secs = started.elapsed().as_secs_f64();
    println!("sentences   {}", counts.sentences);
    println!("skipped     {skipped}");
    println!("precision   {:.2}", counts.precision());
    println!("recall      {:.2}", counts.recall());
    println!("F1          {:.2}", counts.f1());
    println!("exact match {:.2}", counts.exact_match());
    println!("tag acc     {:.2}", counts.tag_accuracy());
    println!("time        {:.1} s ({:.3} s per sentence)", secs, secs / counts.sentences.max(1) as f64);
    Ok(())
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
    if let Some(file) = &args.eval {
        let mut parser = model::Parser::load(&args.model).context("loading model")?;
        return evaluate(&mut parser, file, args.limit);
    }
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
