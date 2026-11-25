// Does a language model collect a character's biography better from the
// text as it is, from the text with pronouns spelled out, from the text
// with more pronouns, from only the sentences that mention the character,
// or from the entity index alone? And at what cost in tokens?
//
// The model is asked to list every fact the text states about one
// character, one per line. Two judgements follow, by the same model:
// for every gold fact, whether the list contains it (recall); for every
// listed statement, whether the original text states it (precision).
//
//   cargo run --release --example story_test -- --model MODEL --text story.txt \
//       --character "Dr. Roylott" --gold roylott.facts
//
// Conditions:
//   raw            the text as it is
//   plain          tokenized and joined back
//   canonical      resolved pronouns with their antecedent in brackets
//   pronominalize  repeated names turned into pronouns
//   mentions       only the sentences that mention the character (by the
//                  entity index, pronouns included), original wording
//   index          no text: the character's descriptors and the skeleton
//                  events it takes part in
//
// Endpoint and cache as in llm_test (LLM_BASE_URL, LLM_API_KEY).

use anyhow::{Context, Result, bail};
use benepar_rs::mentions::{self, Kind};
use benepar_rs::{entities, model, pronominalize, skeleton, text};
use clap::Parser as _;
use serde::Serialize;
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
    /// The story, plain text, one paragraph per line
    #[arg(long)]
    text: PathBuf,
    /// The character, as named in the text
    #[arg(long)]
    character: String,
    /// Gold facts about the character, one per line; lines starting
    /// with # are ignored
    #[arg(long)]
    gold: PathBuf,
    /// Comma-separated conditions
    #[arg(long, default_value = "raw,plain,canonical,pronominalize,mentions,index")]
    conditions: String,
    /// Margin for canonical brackets and pronominalize replacements
    #[arg(long, default_value_t = 0.0)]
    min_margin: f32,
    /// For canonical: below the margin, list up to this many candidates
    #[arg(long, default_value_t = 1)]
    alternatives: usize,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, default_value_t = 60)]
    rpm: u64,
    #[arg(long)]
    reasoning_effort: Option<String>,
    #[arg(long, default_value = "llm-cache")]
    cache: PathBuf,
    /// Write the texts of every condition and the model's lists here
    #[arg(long)]
    out_dir: Option<PathBuf>,
    /// Print the texts and stop before asking
    #[arg(long)]
    dry_run: bool,
}

struct Prepared {
    sents: Vec<text::Sentence>,
    // paragraph of every sentence
    para: Vec<usize>,
    trees: Vec<benepar_rs::Node>,
    all: Vec<mentions::Mention>,
    events: Vec<Vec<skeleton::Event>>,
    offsets: Vec<Vec<(usize, usize)>>,
    paragraphs: Vec<String>,
}

fn prepare(parser: &mut model::Parser, paragraphs: &[String], threads: usize) -> Result<Prepared> {
    let mut sents = Vec::new();
    let mut para = Vec::new();
    for (pi, p) in paragraphs.iter().enumerate() {
        let s = text::sentences(p);
        para.extend(std::iter::repeat_n(pi, s.len()));
        sents.extend(s);
    }
    eprintln!("parsing {} sentences", sents.len());
    let mut trees = Vec::new();
    for chunk in sents.chunks(200) {
        for r in parser.parse_sentences_lenient(chunk, 1, 1, threads) {
            match r {
                Ok(p) => trees.push(p.trees[0].1.clone()),
                Err(e) => bail!("parse failed: {e}"),
            }
        }
    }
    let mut all = Vec::new();
    let mut events = Vec::new();
    for (i, t) in trees.iter().enumerate() {
        let m = mentions::mentions(t, i);
        events.push(skeleton::events(t, i, &m));
        all.extend(m);
    }
    mentions::resolve(&mut all);
    // offsets are per paragraph: attach them paragraph by paragraph
    let mut offsets = Vec::new();
    let mut i = 0;
    for (pi, p) in paragraphs.iter().enumerate() {
        let n = para.iter().filter(|&&x| x == pi).count();
        offsets.extend(text::offsets(p, &sents[i..i + n]));
        i += n;
    }
    mentions::attach_spans(&mut all, &offsets);
    Ok(Prepared { sents, para, trees, all, events, offsets, paragraphs: paragraphs.to_vec() })
}

// Lines joined into paragraphs, as the model will see them.
fn by_paragraph(p: &Prepared, lines: &[String]) -> String {
    let mut out: Vec<Vec<&str>> = vec![Vec::new(); p.paragraphs.len()];
    for (i, l) in lines.iter().enumerate() {
        out[p.para[i]].push(l);
    }
    out.iter().filter(|v| !v.is_empty()).map(|v| v.join(" ")).collect::<Vec<_>>().join("\n")
}

fn render(cond: &str, p: &Prepared, character: &str, min_margin: f32, alternatives: usize) -> Result<String> {
    let words: Vec<Vec<String>> = p.sents.iter().map(|s| s.words.clone()).collect();
    Ok(match cond {
        "raw" => p.paragraphs.join("\n"),
        "plain" => by_paragraph(p, &words.iter().map(|w| mentions::join_words(w)).collect::<Vec<_>>()),
        "canonical" => by_paragraph(p, &mentions::annotate(&words, &p.all, min_margin, alternatives)),
        "pronominalize" => by_paragraph(p, &pronominalize::pronominalize(&p.trees, min_margin).lines()),
        "mentions" | "index" => {
            let ents = entities::entities(&p.trees, &p.all, &p.events);
            let ent = ents
                .iter()
                .filter(|e| e.kind == Kind::Name && mentions::same_name(&e.name, character))
                .max_by_key(|e| e.mentions.len())
                .with_context(|| format!("no entity named like {character:?}; entities: {}", ents.iter().take(20).map(|e| e.name.clone()).collect::<Vec<_>>().join(", ")))?;
            if cond == "mentions" {
                let mut keep: Vec<usize> = ent.mentions.iter().map(|m| m.sentence).collect();
                keep.sort();
                keep.dedup();
                let lines: Vec<String> = (0..p.sents.len())
                    .map(|i| {
                        if !keep.contains(&i) {
                            return String::new();
                        }
                        let o = &p.offsets[i];
                        let (a, b) = (o.first().map_or(0, |x| x.0), o.last().map_or(0, |x| x.1));
                        p.paragraphs[p.para[i]][a..b].to_string()
                    })
                    .collect();
                by_paragraph(p, &lines)
            } else {
                let mut s = format!("Entity: {}\n", ent.name);
                let forms: Vec<&str> = ent.mentions.iter().map(|m| m.text.as_str()).filter(|t| !t.eq_ignore_ascii_case(&ent.name)).collect();
                let mut seen = Vec::new();
                for f in forms {
                    if !seen.iter().any(|x: &String| x.eq_ignore_ascii_case(f)) && !mentions::is_pronoun_word(f) {
                        seen.push(f.to_string());
                    }
                }
                if !seen.is_empty() {
                    s.push_str(&format!("Also called: {}\n", seen.join(", ")));
                }
                for d in &ent.descriptors {
                    s.push_str(&format!("{}: {}\n", d.source, d.text));
                }
                s.push_str("Events:\n");
                for r in &ent.roles {
                    let ev = &p.events[r.sentence][r.event];
                    let line = skeleton::lines(std::slice::from_ref(ev)).remove(0);
                    // "e1 " prefix is meaningless here
                    s.push_str(&format!("s{} {}\n", r.sentence, line.trim_start_matches("e1 ")));
                }
                s
            }
        }
        other => bail!("unknown condition {other}"),
    })
}

fn extract_prompt(text: &str, character: &str) -> String {
    format!(
        "Below is a text, or notes extracted from it. List every fact it gives about {character}: \
who they are, what they did, where, with whom, what they are like. One fact per line, short, \
no numbering, no commentary. Only what the text says.\n\n{text}"
    )
}

fn recall_prompt(list: &str, fact: &str) -> String {
    format!(
        "Here is a list of statements about a character:\n\n{list}\n\nDoes the list contain this fact, \
in any wording? Fact: {fact}\nAnswer yes or no."
    )
}

fn precision_prompt(text: &str, statement: &str) -> String {
    format!("Here is a text:\n\n{text}\n\nDoes the text state or clearly imply the following? {statement}\nAnswer yes or no.")
}

fn yes(reply: &str) -> bool {
    reply.trim().to_lowercase().starts_with("yes")
}

#[derive(Serialize)]
struct Row<'a> {
    condition: &'a str,
    statements: usize,
    gold: usize,
    found: usize,
    supported: usize,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let text_data = std::fs::read_to_string(&args.text)?;
    let paragraphs: Vec<String> = text_data.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
    let gold: Vec<String> = std::fs::read_to_string(&args.gold)?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect();
    let conditions: Vec<&str> = args.conditions.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    let mut parser = model::Parser::load(&args.model_dir).context("loading model")?;
    let p = prepare(&mut parser, &paragraphs, args.threads)?;
    if let Some(d) = &args.out_dir {
        std::fs::create_dir_all(d)?;
    }
    let mut texts = Vec::new();
    for c in &conditions {
        let t = render(c, &p, &args.character, args.min_margin, args.alternatives)?;
        if let Some(d) = &args.out_dir {
            std::fs::write(d.join(format!("{c}.txt")), &t)?;
        }
        eprintln!("{c}: {} chars", t.len());
        texts.push((*c, t));
    }
    if args.dry_run {
        for (c, t) in &texts {
            println!("=== {c}\n{}\n", t.chars().take(1500).collect::<String>());
        }
        return Ok(());
    }
    let mut client = Client::from_env(&args.model, args.reasoning_effort.clone(), &args.cache, args.rpm, 4096)?;
    let raw = paragraphs.join("\n");
    println!("character {}  gold facts {}  model {}", args.character, gold.len(), args.model);
    println!("condition      stmts  recall          precision       prompt  completion");
    let mut rows = Vec::new();
    for (c, t) in &texts {
        let reply = client.ask(&extract_prompt(t, &args.character))?;
        let statements: Vec<String> = reply
            .content
            .lines()
            .map(|l| l.trim().trim_start_matches(['-', '*', ' ']).trim_start_matches(|ch: char| ch.is_ascii_digit() || ch == '.' || ch == ')').trim().to_string())
            .filter(|l| l.len() > 3)
            .collect();
        let list = statements.join("\n");
        if let Some(d) = &args.out_dir {
            std::fs::write(d.join(format!("{c}.list.txt")), &list)?;
        }
        let mut found = 0;
        let mut missing = Vec::new();
        for g in &gold {
            if yes(&client.ask(&recall_prompt(&list, g))?.content) {
                found += 1;
            } else {
                missing.push(g.clone());
            }
        }
        let mut supported = 0;
        let mut unsupported = Vec::new();
        for st in &statements {
            if yes(&client.ask(&precision_prompt(&raw, st))?.content) {
                supported += 1;
            } else {
                unsupported.push(st.clone());
            }
        }
        if let Some(d) = &args.out_dir {
            let mut f = std::fs::File::create(d.join(format!("{c}.judge.txt")))?;
            writeln!(f, "missing gold facts:")?;
            for m in &missing {
                writeln!(f, "  {m}")?;
            }
            writeln!(f, "unsupported statements:")?;
            for u in &unsupported {
                writeln!(f, "  {u}")?;
            }
        }
        let n = statements.len();
        println!(
            "{c:14} {n:5}  {found:3}/{:<3} {:5.1}  {supported:3}/{:<3} {:5.1}  {:>6}  {:>6}",
            gold.len(),
            100.0 * found as f64 / gold.len().max(1) as f64,
            n,
            100.0 * supported as f64 / n.max(1) as f64,
            reply.prompt_tokens.map_or("-".to_string(), |x| x.to_string()),
            reply.completion_tokens.map_or("-".to_string(), |x| x.to_string())
        );
        rows.push(Row { condition: c, statements: n, gold: gold.len(), found, supported, prompt_tokens: reply.prompt_tokens, completion_tokens: reply.completion_tokens });
    }
    if let Some(d) = &args.out_dir {
        std::fs::write(d.join("summary.json"), serde_json::to_string_pretty(&rows)?)?;
    }
    Ok(())
}
