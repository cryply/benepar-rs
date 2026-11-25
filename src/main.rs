use benepar_rs::{entities, gap, mentions, model, pronominalize, skeleton, text, treebank};
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
    /// Score the sentence skeleton (subject and object of every verb)
    /// against the function tags of a treebank file or directory
    #[arg(long, value_name = "FILE")]
    eval_skeleton: Option<PathBuf>,
    /// Score the pronoun rules on a GAP .tsv file (gap-development.tsv,
    /// gap-test.tsv) and print GAP F1 by gender, plus precision by margin
    #[arg(long, value_name = "FILE")]
    gap: Option<PathBuf>,
    /// With --eval: stop after this many sentences; with --gap: after this
    /// many examples
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
    /// Print the text with repeated mentions replaced by pronouns where
    /// the pronoun still resolves to the same entity; with --json, one
    /// object with the lines and the list of replacements
    #[arg(long)]
    pronominalize: bool,
    /// Print the skeleton of every sentence: each verb with its subject,
    /// objects, prepositional and clause arguments; with --json, the
    /// events with word positions and mention indices
    #[arg(long)]
    skeleton: bool,
    /// Print the entities of the text: each with its mentions, what the
    /// text says about it (titles, modifiers, appositions, complements)
    /// and its roles in the skeleton's events; --json for the full index
    #[arg(long)]
    entities: bool,
    /// With --canonical: annotate a pronoun only if its best antecedent is
    /// at least this far ahead of the next one that is not the same name;
    /// with --pronominalize: insert a pronoun only with this margin
    #[arg(long, default_value_t = 0.0)]
    min_margin: f32,
    /// With --canonical: when the margin is below --min-margin, list up to
    /// this many candidates instead of nothing ("he [Smith or Jones]")
    #[arg(long, default_value_t = 1)]
    alternatives: usize,
    /// Print the hyphenated compounds of every sentence as JSON: parts,
    /// tags, pattern and the role of the left part
    #[arg(long)]
    compounds: bool,
    /// Keep hyphenated words whole ("owner-controlled") instead of
    /// splitting them at the hyphen as the treebank does
    #[arg(long)]
    keep_hyphens: bool,
    /// Text to parse; read from stdin when missing
    text: Option<String>,
}

// Trees of a file, or of every .mrg/.tree file in a directory (an LDC
// section such as penntree/23), read in name order.
fn read_trees(file: &std::path::Path) -> Result<Vec<benepar_rs::Node>> {
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
    Ok(raw)
}

fn evaluate(parser: &mut model::Parser, file: &std::path::Path, limit: Option<usize>, threads: usize) -> Result<()> {
    let raw = read_trees(file)?;
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

fn evaluate_skeleton(parser: &mut model::Parser, file: &std::path::Path, limit: Option<usize>, threads: usize) -> Result<()> {
    let raw = read_trees(file)?;
    let mut golds = Vec::new();
    let mut sents = Vec::new();
    for (n, tree) in raw.iter().enumerate() {
        if limit.is_some_and(|l| n >= l) {
            break;
        }
        let Some(clean) = treebank::clean(tree) else { continue };
        let words: Vec<String> = treebank::leaves(&clean).iter().map(|(_, w)| treebank::unescape(w)).collect();
        let space_after = treebank::guess_spaces(&words);
        sents.push(text::Sentence { words, space_after });
        golds.push((skeleton::gold_relations(tree), clean));
    }
    let started = std::time::Instant::now();
    let mut ours = skeleton::Counts::default();
    // the same rules on the gold trees: how much is lost in parsing
    let mut on_gold = skeleton::Counts::default();
    let mut skipped = 0usize;
    for (ci, chunk) in sents.chunks(200).enumerate() {
        let results = parser.parse_sentences_lenient(chunk, 1, 1, threads);
        for (i, r) in results.into_iter().enumerate() {
            let n = ci * 200 + i;
            let (gold, clean) = &golds[n];
            match r {
                Ok(p) => {
                    let ev = skeleton::events(&p.trees[0].1, 0, &[]);
                    ours.add(gold, &skeleton::relations(&ev));
                    let ev = skeleton::events(clean, 0, &[]);
                    on_gold.add(gold, &skeleton::relations(&ev));
                }
                Err(e) => {
                    eprintln!("line {}: {e}", n + 1);
                    skipped += 1;
                }
            }
        }
        eprintln!("{} sentences", (ci + 1) * 200);
    }
    println!("sentences {}  skipped {skipped}", golds.len());
    println!("relation  trees        P      R      F1    gold  pred");
    for (name, c) in [("our parses", &ours), ("gold trees", &on_gold)] {
        let (p, r, f) = skeleton::Counts::f1(c.subj_match, c.subj_gold, c.subj_pred);
        println!("subject   {name}  {p:5.1}  {r:5.1}  {f:5.1}  {:5}  {:5}", c.subj_gold, c.subj_pred);
        let (p, r, f) = skeleton::Counts::f1(c.obj_match, c.obj_gold, c.obj_pred);
        println!("object    {name}  {p:5.1}  {r:5.1}  {f:5.1}  {:5}  {:5}", c.obj_gold, c.obj_pred);
    }
    println!("time      {:.1} s", started.elapsed().as_secs_f64());
    Ok(())
}

fn evaluate_gap(parser: &mut model::Parser, file: &std::path::Path, args: &Args) -> Result<()> {
    let data = std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
    let mut examples = gap::parse_tsv(&data)?;
    if let Some(l) = args.limit {
        examples.truncate(l);
    }
    let split = |t: &str| if args.keep_hyphens { text::sentences_keep_hyphens(t) } else { text::sentences(t) };
    // All sentences of all examples go through the parser together so
    // that --threads helps; `ranges` maps them back.
    let mut sents = Vec::new();
    let mut ranges = Vec::new();
    for ex in &examples {
        let s = split(&ex.text);
        ranges.push(sents.len()..sents.len() + s.len());
        sents.extend(s);
    }
    let started = std::time::Instant::now();
    let parses = parser.parse_sentences_lenient(&sents, 1, 1, args.threads);
    let mut report = gap::Report::default();
    let mut failed = 0usize;
    for (ex, range) in examples.iter().zip(ranges) {
        let mut all = Vec::new();
        let mut ok = true;
        for (i, r) in parses[range.clone()].iter().enumerate() {
            match r {
                Ok(p) => all.extend(mentions::mentions(&p.trees[0].1, i)),
                Err(e) => {
                    eprintln!("{}: {e}", ex.id);
                    ok = false;
                }
            }
        }
        if !ok {
            failed += 1;
        }
        mentions::resolve(&mut all);
        mentions::attach_spans(&mut all, &text::offsets(&ex.text, &sents[range]));
        report.add(ex, &gap::predict(ex, &all));
    }
    report.print();
    println!("parse errors  {failed}");
    println!("time          {:.1} s", started.elapsed().as_secs_f64());
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
    if let Some(file) = &args.eval_skeleton {
        let mut parser = model::Parser::load(&args.model).context("loading model")?;
        return evaluate_skeleton(&mut parser, file, args.limit, args.threads);
    }
    if let Some(file) = &args.gap {
        let mut parser = model::Parser::load(&args.model).context("loading model")?;
        return evaluate_gap(&mut parser, file, &args);
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
    if args.pronominalize {
        let trees: Vec<_> = parses.iter().map(|p| p.trees[0].1.clone()).collect();
        let r = pronominalize::pronominalize(&trees, args.min_margin);
        if args.json {
            println!("{}", serde_json::json!({"lines": r.lines(), "replacements": r.replacements}));
        } else {
            for line in r.lines() {
                println!("{line}");
            }
        }
        return Ok(());
    }
    if args.skeleton {
        let mut out = Vec::new();
        for (i, p) in parses.iter().enumerate() {
            let m = mentions::mentions(&p.trees[0].1, i);
            let ev = skeleton::events(&p.trees[0].1, i, &m);
            if args.json {
                out.push(serde_json::json!({"sentence": i, "events": ev, "mentions": m}));
            } else {
                for line in skeleton::lines(&ev) {
                    println!("s{i} {line}");
                }
            }
        }
        if args.json {
            println!("{}", serde_json::Value::Array(out));
        }
        return Ok(());
    }
    if args.entities {
        let trees: Vec<_> = parses.iter().map(|p| p.trees[0].1.clone()).collect();
        let mut all = Vec::new();
        let mut events = Vec::new();
        for (i, t) in trees.iter().enumerate() {
            let m = mentions::mentions(t, i);
            events.push(skeleton::events(t, i, &m));
            all.extend(m);
        }
        mentions::resolve(&mut all);
        mentions::attach_spans(&mut all, &text::offsets(&text, &sents));
        let ents = entities::entities(&trees, &all, &events);
        if args.json {
            println!("{}", serde_json::json!({"entities": ents, "events": events}));
        } else {
            for line in entities::lines(&ents) {
                println!("{line}");
            }
        }
        return Ok(());
    }
    if args.mentions || args.canonical {
        let mut all = Vec::new();
        for (i, p) in parses.iter().enumerate() {
            all.extend(mentions::mentions(&p.trees[0].1, i));
        }
        mentions::resolve(&mut all);
        mentions::attach_spans(&mut all, &text::offsets(&text, &sents));
        if args.canonical {
            let words: Vec<Vec<String>> = parses.iter().map(|p| p.words.clone()).collect();
            for line in mentions::annotate(&words, &all, args.min_margin, args.alternatives) {
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
    for (i, parse) in parses.iter().enumerate() {
        if args.compounds {
            for c in parse.compounds() {
                let mut v = serde_json::to_value(&c)?;
                v["sentence"] = serde_json::json!(i);
                println!("{v}");
            }
            continue;
        }
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
