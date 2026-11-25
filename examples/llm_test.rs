// Does a language model read pronouns better when the text spells out
// what they refer to (--canonical), or when repeated names are turned into
// pronouns (--pronominalize)? This asks one model the same question about
// the same examples in several forms of the text and compares accuracy.
//
// Data: GAP (gap-development.tsv, gap-test.tsv) or the Winograd Schema
// Collection (WSCollection.xml; its first 273 schemas are WSC273). Neither
// is part of the repository.
//
// The model is any OpenAI-compatible chat endpoint:
//   LLM_BASE_URL  (default https://openrouter.ai/api/v1; Ollama: http://HOST:11434/v1)
//   LLM_API_KEY   (not needed by Ollama)
// Responses are cached on disk, so a rerun costs nothing and a run cut
// short by a rate limit picks up where it stopped.
//
//   cargo run --release --example llm_test -- --model MODEL --gap gap-development.tsv --limit 300
//   cargo run --release --example llm_test -- --model MODEL --wsc WSCollection.xml

use anyhow::{Context, Result, bail};
use benepar_rs::mentions::{self, Kind};
use benepar_rs::{gap, model, pronominalize, text};
use clap::Parser as _;
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

mod common;
use common::Client;

#[derive(clap::Parser)]
struct Args {
    /// Directory with the benepar model files
    #[arg(long, env = "BENEPAR_MODEL")]
    model_dir: PathBuf,
    /// Chat model name as the endpoint knows it
    #[arg(long)]
    model: String,
    /// GAP .tsv file
    #[arg(long, value_name = "FILE")]
    gap: Option<PathBuf>,
    /// Winograd Schema Collection .xml file
    #[arg(long, value_name = "FILE")]
    wsc: Option<PathBuf>,
    /// Stop after this many examples (WSC273: 273)
    #[arg(long)]
    limit: Option<usize>,
    /// Comma-separated conditions: raw, plain, canonical, pronominalize
    #[arg(long, default_value = "raw,plain,canonical,pronominalize")]
    conditions: String,
    /// Margin for --canonical brackets and --pronominalize replacements
    #[arg(long, default_value_t = 0.0)]
    min_margin: f32,
    /// For canonical: below the margin, list up to this many candidates
    #[arg(long, default_value_t = 1)]
    alternatives: usize,
    /// Parser threads
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Requests per minute to the model
    #[arg(long, default_value_t = 20)]
    rpm: u64,
    /// reasoning_effort for models that think before answering (low,
    /// medium, high); not sent when absent
    #[arg(long)]
    reasoning_effort: Option<String>,
    /// Directory for cached model responses
    #[arg(long, default_value = "llm-cache")]
    cache: PathBuf,
    /// Write one JSON line per example and condition here
    #[arg(long)]
    out: Option<PathBuf>,
    /// Print the prompts of the first examples and stop before asking
    #[arg(long)]
    dry_run: bool,
}

struct Example {
    id: String,
    text: String,
    // byte offset of the pronoun in `text`
    pronoun: usize,
    options: Vec<String>,
    // index into options
    gold: usize,
    // the GAP record, for matching our own antecedent to A and B
    gap: Option<gap::Example>,
}

// Minimal reader for the Winograd Schema Collection XML: the file is small
// and regular, so tags are found by name.
fn wsc_examples(data: &str) -> Result<Vec<Example>> {
    fn inner<'a>(s: &'a str, tag: &str) -> Option<(&'a str, &'a str)> {
        let open = format!("<{tag}>");
        let close = format!("</{tag}>");
        let a = s.find(&open)? + open.len();
        let b = a + s[a..].find(&close)?;
        Some((&s[a..b], &s[b + close.len()..]))
    }
    fn clean(s: &str) -> String {
        let s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'");
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }
    let mut out = Vec::new();
    let mut rest = data;
    let mut n = 0;
    while let Some((schema, after)) = inner(rest, "schema") {
        rest = after;
        n += 1;
        let (textpart, _) = inner(schema, "text").context("schema without text")?;
        let (t1, r) = inner(textpart, "txt1").context("txt1")?;
        let (pron, r) = inner(r, "pron").context("pron")?;
        let (t2, _) = inner(r, "txt2").context("txt2")?;
        let (t1, pron, t2) = (clean(t1), clean(pron), clean(t2));
        let mut text = t1.clone();
        if !text.is_empty() {
            text.push(' ');
        }
        let pronoun = text.len();
        text.push_str(&pron);
        if !t2.is_empty() {
            if !t2.starts_with(|c: char| c.is_ascii_punctuation() && c != '"' && c != '(') {
                text.push(' ');
            }
            text.push_str(&t2);
        }
        let (answers, _) = inner(schema, "answers").context("answers")?;
        let mut options = Vec::new();
        let mut r = answers;
        while let Some((a, after)) = inner(r, "answer") {
            options.push(clean(a));
            r = after;
        }
        if options.len() != 2 {
            bail!("schema {n}: {} answers", options.len());
        }
        let (correct, _) = inner(schema, "correctAnswer").context("correctAnswer")?;
        let gold = match clean(correct).trim_end_matches('.').to_uppercase().as_str() {
            "A" => 0,
            "B" => 1,
            other => bail!("schema {n}: correct answer {other:?}"),
        };
        out.push(Example { id: format!("wsc-{n}"), text, pronoun, options, gold, gap: None });
    }
    Ok(out)
}

fn gap_examples(data: &str) -> Result<Vec<Example>> {
    Ok(gap::parse_tsv(data)?
        .into_iter()
        .map(|ex| {
            let gold = match (ex.a.coref, ex.b.coref) {
                (true, _) => 0,
                (_, true) => 1,
                _ => 2,
            };
            Example {
                id: ex.id.clone(),
                text: ex.text.clone(),
                pronoun: ex.pronoun_offset,
                options: vec![ex.a.text.clone(), ex.b.text.clone(), "neither A nor B".to_string()],
                gold,
                gap: Some(ex),
            }
        })
        .collect())
}

// The sentences of one example, parsed, with the pronoun located.
struct Parsed {
    sents: Vec<text::Sentence>,
    trees: Vec<benepar_rs::Node>,
    // (sentence, word index) of the pronoun
    at: Option<(usize, usize)>,
}

fn join(words: &[String]) -> String {
    words.join(" ").replace(" - ", "-")
}

fn mark(w: &str) -> String {
    format!("[[{w}]]")
}

// The text of one condition with the pronoun marked, or None when the
// condition cannot be built for this example. For canonical, also which
// option our own bracket on the marked pronoun points to (2 = neither),
// or None without a bracket.
fn render(cond: &str, ex: &Example, p: &Parsed, min_margin: f32, alternatives: usize) -> Option<(String, Option<usize>)> {
    let mut hint = None;
    if cond == "raw" {
        let end = ex.pronoun + ex.text[ex.pronoun..].find(|c: char| !c.is_alphanumeric()).unwrap_or(ex.text.len() - ex.pronoun);
        return Some((format!("{}{}{}", &ex.text[..ex.pronoun], mark(&ex.text[ex.pronoun..end]), &ex.text[end..]), None));
    }
    let (ps, pw) = p.at?;
    let mut lines: Vec<Vec<String>> = p.sents.iter().map(|s| s.words.clone()).collect();
    match cond {
        "plain" => {
            lines[ps][pw] = mark(&lines[ps][pw]);
        }
        "canonical" => {
            let mut all = Vec::new();
            for (i, t) in p.trees.iter().enumerate() {
                all.extend(mentions::mentions(t, i));
            }
            mentions::resolve(&mut all);
            mentions::attach_spans(&mut all, &text::offsets(&ex.text, &p.sents));
            lines[ps][pw] = mark(&lines[ps][pw]);
            for m in all.iter().filter(|m| m.kind == Kind::Pronoun) {
                let Some(b) = mentions::bracket(m, min_margin, alternatives) else { continue };
                let Some(c) = m.candidates.first() else { continue };
                {
                    lines[m.sentence][m.start] = format!("{} [{}]", lines[m.sentence][m.start], b);
                    if (m.sentence, m.start) == (ps, pw) && !b.contains(" or ") {
                        hint = Some(match ex.gap.as_ref().and_then(|g| gap::slot_of(g, &c.text, c.span)) {
                            Some(gap::Slot::A) => 0,
                            Some(gap::Slot::B) => 1,
                            None => 2,
                        });
                    }
                }
            }
        }
        "pronominalize" => {
            let r = pronominalize::pronominalize(&p.trees, min_margin);
            lines = r.trees.iter().map(|t| benepar_rs::treebank::leaves(t).into_iter().map(|(_, w)| w).collect()).collect();
            let j = r.origin[ps].iter().position(|&o| o == pw)?;
            lines[ps][j] = mark(&lines[ps][j]);
        }
        _ => return None,
    }
    Some((lines.iter().map(|l| join(l)).collect::<Vec<_>>().join(" "), hint))
}

fn prompt(text: &str, options: &[String]) -> String {
    let mut s = String::new();
    s.push_str("Read the text and answer the question.\n\nText: ");
    s.push_str(text);
    s.push_str("\n\nQuestion: In the text, who or what does the pronoun marked with [[ ]] refer to?\n");
    for (i, o) in options.iter().enumerate() {
        s.push_str(&format!("{}) {}\n", (b'A' + i as u8) as char, o));
    }
    s.push_str("Answer with the letter only.");
    s
}

// The letter the model chose, if any.
fn parse_answer(reply: &str, n: usize) -> Option<usize> {
    let t = reply.trim();
    let lower = t.to_lowercase();
    if n > 2 && lower.starts_with("neither") {
        return Some(2);
    }
    let first = t.trim_start_matches(|c: char| !c.is_ascii_alphabetic()).chars().next()?;
    let i = (first.to_ascii_uppercase() as u8).checked_sub(b'A')? as usize;
    let next = t.trim_start_matches(|c: char| !c.is_ascii_alphabetic()).chars().nth(1);
    if i < n && !next.is_some_and(|c| c.is_ascii_alphabetic()) { Some(i) } else { None }
}

#[derive(Default)]
struct Tally {
    n: usize,
    right: usize,
    invalid: usize,
    tokens: u64,
    with_tokens: usize,
    // per example id: answered right
    results: BTreeMap<String, bool>,
}

fn wilson(k: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let z = 1.96f64;
    let (k, n) = (k as f64, n as f64);
    let p = k / n;
    let d = 1.0 + z * z / n;
    let c = (p + z * z / (2.0 * n)) / d;
    let h = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / d;
    (c - h, c + h)
}

// Two-sided exact McNemar test: b examples right in the first condition
// only, c right in the second only.
fn mcnemar(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let ln_fact = |m: usize| (1..=m).map(|i| (i as f64).ln()).sum::<f64>();
    let pmf = |k: usize| (ln_fact(n) - ln_fact(k) - ln_fact(n - k) - (n as f64) * 2f64.ln()).exp();
    let k = b.min(c);
    let tail: f64 = (0..=k).map(pmf).sum();
    (2.0 * tail).min(1.0)
}

#[derive(Serialize)]
struct Record<'a> {
    id: &'a str,
    condition: &'a str,
    gold: usize,
    answer: Option<usize>,
    correct: bool,
    // canonical only: the option our bracket on the pronoun points to
    hint: Option<usize>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    reply: &'a str,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut examples = match (&args.gap, &args.wsc) {
        (Some(f), None) => gap_examples(&std::fs::read_to_string(f)?)?,
        (None, Some(f)) => {
            let bytes = std::fs::read(f)?;
            // the collection is ISO-8859-1
            let s: String = bytes.iter().map(|&b| b as char).collect();
            wsc_examples(&s)?
        }
        _ => bail!("give exactly one of --gap and --wsc"),
    };
    if let Some(l) = args.limit {
        examples.truncate(l);
    }
    let conditions: Vec<&str> = args.conditions.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    for c in &conditions {
        if !matches!(*c, "raw" | "plain" | "canonical" | "pronominalize") {
            bail!("unknown condition {c}");
        }
    }

    eprintln!("parsing {} examples", examples.len());
    let mut parser = model::Parser::load(&args.model_dir).context("loading model")?;
    let mut sents = Vec::new();
    let mut ranges = Vec::new();
    for ex in &examples {
        let s = text::sentences(&ex.text);
        ranges.push(sents.len()..sents.len() + s.len());
        sents.extend(s);
    }
    let parses = parser.parse_sentences_lenient(&sents, 1, 1, args.threads);
    let mut parsed = Vec::new();
    for (ex, range) in examples.iter().zip(ranges) {
        let mut trees = Vec::new();
        for r in &parses[range.clone()] {
            match r {
                Ok(p) => trees.push(p.trees[0].1.clone()),
                Err(e) => {
                    eprintln!("{}: {e}", ex.id);
                    break;
                }
            }
        }
        let sents = sents[range].to_vec();
        let offs = text::offsets(&ex.text, &sents);
        let mut at = None;
        for (si, o) in offs.iter().enumerate() {
            if let Some(wi) = o.iter().position(|&(a, b)| a <= ex.pronoun && ex.pronoun < b) {
                at = Some((si, wi));
            }
        }
        if trees.len() != sents.len() {
            at = None;
        }
        parsed.push(Parsed { sents, trees, at });
    }

    let mut client = if args.dry_run {
        None
    } else {
        Some(Client::from_env(&args.model, args.reasoning_effort.clone(), &args.cache, args.rpm, 256)?)
    };
    let mut out = match &args.out {
        Some(p) => Some(std::fs::File::create(p)?),
        None => None,
    };

    let mut tallies: BTreeMap<&str, Tally> = BTreeMap::new();
    let mut skipped = 0;
    for (i, (ex, p)) in examples.iter().zip(&parsed).enumerate() {
        let mut texts = Vec::new();
        for c in &conditions {
            match render(c, ex, p, args.min_margin, args.alternatives) {
                Some((t, h)) => texts.push((*c, t, h)),
                None => {
                    texts.clear();
                    break;
                }
            }
        }
        if texts.is_empty() {
            skipped += 1;
            continue;
        }
        if args.dry_run {
            if i < 3 {
                for (c, t, _) in &texts {
                    println!("--- {} {c}\n{}\n", ex.id, prompt(t, &ex.options));
                }
                println!("gold: {}", (b'A' + ex.gold as u8) as char);
            }
            continue;
        }
        for (c, t, hint) in &texts {
            let pr = prompt(t, &ex.options);
            let reply = client.as_mut().expect("client").ask(&pr)?;
            let answer = parse_answer(&reply.content, ex.options.len());
            let correct = answer == Some(ex.gold);
            let tally = tallies.entry(c).or_default();
            tally.n += 1;
            tally.right += correct as usize;
            tally.invalid += answer.is_none() as usize;
            if let Some(k) = reply.prompt_tokens {
                tally.tokens += k;
                tally.with_tokens += 1;
            }
            tally.results.insert(ex.id.clone(), correct);
            if let Some(f) = out.as_mut() {
                let rec = Record { id: &ex.id, condition: c, gold: ex.gold, answer, correct, hint: *hint, prompt_tokens: reply.prompt_tokens, completion_tokens: reply.completion_tokens, reply: reply.content.trim() };
                writeln!(f, "{}", serde_json::to_string(&rec)?)?;
            }
        }
        if (i + 1) % 25 == 0 {
            eprintln!("{} examples done", i + 1);
        }
    }
    if args.dry_run {
        eprintln!("dry run: {} examples, {skipped} skipped", examples.len() - skipped);
        return Ok(());
    }

    println!("model {}", args.model);
    println!("examples {} (skipped {skipped}: pronoun not located after tokenization)", examples.len() - skipped);
    println!("condition       n   acc    95% CI          tokens  invalid");
    for c in &conditions {
        let Some(t) = tallies.get(c) else { continue };
        let acc = 100.0 * t.right as f64 / t.n.max(1) as f64;
        let (lo, hi) = wilson(t.right, t.n);
        let tok = if t.with_tokens > 0 { format!("{:6.0}", t.tokens as f64 / t.with_tokens as f64) } else { "     -".to_string() };
        println!("{c:14} {:4}  {acc:5.1}  [{:5.1}, {:5.1}]  {tok}  {}", t.n, 100.0 * lo, 100.0 * hi, t.invalid);
    }
    if let Some(first) = tallies.get(conditions[0]) {
        println!("against {}:", conditions[0]);
        for c in &conditions[1..] {
            let Some(t) = tallies.get(c) else { continue };
            let mut b = 0;
            let mut cc = 0;
            for (id, &r1) in &first.results {
                let Some(&r2) = t.results.get(id) else { continue };
                b += (r1 && !r2) as usize;
                cc += (!r1 && r2) as usize;
            }
            println!("  {c:14} right only in {}: {b:4}  right only in {c}: {cc:4}  McNemar p = {:.4}", conditions[0], mcnemar(b, cc));
        }
    }
    Ok(())
}
