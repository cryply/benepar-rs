use benepar_rs::{mentions, model, text, treebank};
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
    /// Score the parser on a treebank file or a directory of .mrg/.tree
    /// files and print EVALB-style precision, recall and F1
    #[arg(long, value_name = "FILE")]
    eval: Option<PathBuf>,
    /// With --eval: stop after this many sentences
    #[arg(long)]
    limit: Option<usize>,
    /// Print one JSON object per sentence instead of bracketed trees
    #[arg(long)]
    json: bool,
    /// Parse this many sentences at once on separate threads
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Print the mentions (names, descriptions, pronouns with ranked
    /// antecedents) and the entity index as JSON
    #[arg(long)]
    mentions: bool,
    /// Print the text with every resolved pronoun followed by its best
    /// antecedent in brackets
    #[arg(long)]
    canonical: bool,
    /// Keep hyphenated words whole ("owner-controlled") instead of
    /// splitting them at the hyphen as the treebank does
    #[arg(long)]
    keep_hyphens: bool,
    /// Text to parse; read from stdin when missing
    text: Option<String>,
}

fn evaluate(parser: &mut model::Parser, file: &std::path::Path, limit: Option<usize>, threads: usize) -> Result<()> {
    // A file with trees, or a directory of .mrg/.tree files (an LDC
    // section such as penntree/23), read in name order.
    let mut paths = Vec::new();
    if file.is_dir() {
        for e in std::fs::read_dir(file)? {
            let p = e?.path();
            if matches!(p.extension().and_then(|x| x.to_str()), Some("mrg" | "tree")) {
                paths.push(p);
            }
        }
        paths.sort();
    } else {
        paths.push(file.to_path_buf());
    }
    let mut raw = Vec::new();
    for p in &paths {
        let data = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
        let trees = treebank::parse_many(&data).with_context(|| format!("parsing {}", p.display()))?;
        raw.extend(trees);
    }
    let mut golds = Vec::new();
    let mut sents = Vec::new();
    for (n, tree) in raw.iter().enumerate() {
        if limit.is_some_and(|l| n >= l) {
            break;
        }
        let Some(gold) = treebank::clean(tree) else { continue };
        let words: Vec<String> = treebank::leaves(&gold)
            .iter()
            .map(|(_, w)| treebank::unescape(w))
            .collect();
        let space_after = treebank::guess_spaces(&words);
        sents.push(text::Sentence { words, space_after });
        golds.push(gold);
    }
    let started = std::time::Instant::now();
    let mut counts = treebank::Counts::default();
    let mut skipped = 0usize;
    // Chunks keep memory bounded and give progress output.
    for (ci, chunk) in sents.chunks(200).enumerate() {
        let results = parser.parse_sentences_lenient(chunk, 1, 1, threads);
        for (i, r) in results.into_iter().enumerate() {
            let n = ci * 200 + i;
            match r {
                Ok(p) => counts.add(&golds[n], &p.trees[0].1),
                Err(e) => {
                    eprintln!("line {}: {e}", n + 1);
                    skipped += 1;
                }
            }
        }
        eprintln!("{} sentences, F1 {:.2}", counts.sentences + skipped, counts.f1());
    }
    let secs = started.elapsed().as_secs_f64();
    println!("sentences   {}", counts.sentences);
    println!("skipped     {skipped}");
    println!("precision   {:.2}", counts.precision());
    println!("recall      {:.2}", counts.recall());
    println!("F1          {:.2}", counts.f1());
    println!("exact match {:.2}", counts.exact_match());
    println!("tag acc     {:.2}", counts.tag_accuracy());
    println!("time        {:.1} s ({:.3} s per sentence, {threads} threads)", secs, secs / counts.sentences.max(1) as f64);
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
        return evaluate(&mut parser, file, args.limit, args.threads);
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
    let sents = if args.keep_hyphens { text::sentences_keep_hyphens(&text) } else { text::sentences(&text) };
    let parses = parser.parse_sentences(&sents, args.k, args.label_variants, args.threads)?;
    if args.mentions || args.canonical {
        let mut all = Vec::new();
        for (i, p) in parses.iter().enumerate() {
            all.extend(mentions::mentions(&p.trees[0].1, i));
        }
        mentions::resolve(&mut all);
        if args.canonical {
            let words: Vec<Vec<String>> = parses.iter().map(|p| p.words.clone()).collect();
            for line in mentions::annotate(&words, &all) {
                println!("{line}");
            }
        }
        if args.mentions {
            let ents: Vec<serde_json::Value> = mentions::entities(&all)
                .into_iter()
                .map(|(name, sents)| serde_json::json!({"entity": name, "sentences": sents}))
                .collect();
            println!("{}", serde_json::json!({"mentions": all, "entities": ents}));
        }
        return Ok(());
    }
    for parse in parses {
        if args.json {
            println!("{}", parse.to_json());
            continue;
        }
        for (score, t) in &parse.trees {
            if args.scores {
                println!("{score:.3}\t{t}");
            } else {
                println!("{t}");
            }
        }
        if args.k > 1 {
            println!();
        }
    }
    Ok(())
}
