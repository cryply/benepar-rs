// A gold file of pronoun antecedents for a story, and the resolver's
// score against it.
//
// draft: for every third person pronoun the text has, ask a chat model
// what it refers to, given the sentence and the sentences before it, and
// write one line per pronoun:
//
//   sentence  word  pronoun  antecedent  resolver
//
// where antecedent is the model's answer (words from the text, NONE for
// an "it" that stands for nothing, CLAUSE for one that stands for a
// statement) and resolver is what src/mentions.rs picked. The file is
// meant to be checked by hand, especially the lines where the two
// disagree; the model's answer is a draft, not the truth. A sixth
// column may hold "alt:<words>", a second acceptable answer. Lines
// starting with "# alias:" list the names of one person or thing,
// separated by " = ", so that "Dr. Roylott" and "my stepfather" count
// as the same antecedent:
//
//   # alias: Dr. Grimesby Roylott = Dr. Roylott = my stepfather
//
//   cargo run --release --example pronoun_gold -- draft --model MODEL \
//       --text story.txt --out story.pronouns.tsv
//
// score: read a checked file and report how often the resolver's first
// candidate matches the antecedent, by pronoun word.
//
//   cargo run --release --example pronoun_gold -- score --text story.txt \
//       --gold story.pronouns.tsv
//
// Endpoint and cache as in llm_test (LLM_BASE_URL, LLM_API_KEY).

use anyhow::{Context, Result};
use benepar_rs::mentions::{self, Kind, Mention};
use benepar_rs::{model, text};
use clap::Parser as _;
use std::io::Write;
use std::path::PathBuf;

mod common;
use common::Client;

#[derive(clap::Parser)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Subcommand)]
enum Cmd {
    Draft(Draft),
    Score(Score),
}

#[derive(clap::Args)]
struct Draft {
    #[arg(long, env = "BENEPAR_MODEL")]
    model_dir: PathBuf,
    #[arg(long)]
    model: String,
    #[arg(long)]
    text: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Sentences of context before the pronoun's sentence
    #[arg(long, default_value_t = 8)]
    context: usize,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, default_value_t = 60)]
    rpm: u64,
    #[arg(long)]
    reasoning_effort: Option<String>,
    #[arg(long, default_value = "llm-cache")]
    cache: PathBuf,
}

#[derive(clap::Args)]
struct Score {
    #[arg(long, env = "BENEPAR_MODEL")]
    model_dir: PathBuf,
    #[arg(long)]
    text: PathBuf,
    #[arg(long)]
    gold: PathBuf,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Print every miss
    #[arg(long)]
    verbose: bool,
}

struct Doc {
    sents: Vec<text::Sentence>,
    all: Vec<Mention>,
}

fn parse(model_dir: &PathBuf, text_path: &PathBuf, threads: usize) -> Result<Doc> {
    let data = std::fs::read_to_string(text_path)?;
    let mut sents = Vec::new();
    for p in data.lines().map(str::trim).filter(|l| !l.is_empty()) {
        sents.extend(text::sentences(p));
    }
    let mut parser = model::Parser::load(model_dir).context("loading model")?;
    eprintln!("parsing {} sentences", sents.len());
    let mut all = Vec::new();
    let mut i = 0;
    for chunk in sents.chunks(200) {
        for r in parser.parse_sentences_lenient(chunk, 1, 1, threads) {
            let p = r.map_err(|e| anyhow::anyhow!("parse failed: {e}"))?;
            all.extend(mentions::mentions(&p.trees[0].1, i));
            i += 1;
        }
    }
    mentions::resolve(&mut all);
    Ok(Doc { sents, all })
}

fn third_person(w: &str) -> bool {
    matches!(
        w.to_lowercase().as_str(),
        "he" | "him" | "his" | "himself" | "she" | "her" | "hers" | "herself" | "it" | "its" | "itself" | "they" | "them" | "their" | "theirs" | "themselves"
    )
}

fn marked(doc: &Doc, m: &Mention, context: usize) -> String {
    let from = m.sentence.saturating_sub(context);
    let mut out = String::new();
    for i in from..=m.sentence {
        let mut words = doc.sents[i].words.clone();
        if i == m.sentence {
            words[m.start] = format!("<<{}>>", words[m.start]);
        }
        out.push_str(&mentions::join_words(&words));
        out.push('\n');
    }
    out
}

fn prompt(context: &str, word: &str) -> String {
    format!(
        "In the passage below one pronoun is marked with << >>. Say what it refers to.\n\
Answer with the exact words from the passage that name the person or thing, as short as a name or a noun phrase \
(for example \"Dr. Roylott\", \"my sister\", \"the lamp\"). If the pronoun refers to a whole statement or event \
rather than to a person or thing, answer CLAUSE. If it refers to nothing (as in \"it is clear that\", \
\"it was late\"), answer NONE. Never answer with a pronoun: name the person or thing. \
Give only the answer, no explanation.\n\n{context}\nMarked pronoun: {word}"
    )
}

fn draft(a: Draft) -> Result<()> {
    let doc = parse(&a.model_dir, &a.text, a.threads)?;
    let mut client = Client::from_env(&a.model, a.reasoning_effort.clone(), &a.cache, a.rpm, 512)?;
    let mut f = std::fs::File::create(&a.out)?;
    writeln!(f, "sentence\tword\tpronoun\tantecedent\tresolver")?;
    let mut n = 0;
    for m in doc.all.iter().filter(|m| m.kind == Kind::Pronoun && third_person(&m.text)) {
        let reply = client.ask(&prompt(&marked(&doc, m, a.context), &m.text))?;
        let answer = reply.content.lines().next().unwrap_or("").trim().trim_matches('"').to_string();
        let resolver = m.candidates.first().map(|c| c.text.clone()).unwrap_or_default();
        writeln!(f, "{}\t{}\t{}\t{}\t{}", m.sentence, m.start, m.text, answer, resolver)?;
        n += 1;
        if n % 50 == 0 {
            eprintln!("{n} pronouns");
        }
    }
    eprintln!("{n} pronouns written to {}", a.out.display());
    Ok(())
}

// Does the resolver's pick name the same thing as the gold antecedent.
fn matches(gold: &str, pick: &str) -> bool {
    let g = gold.trim().to_lowercase();
    let p = pick.trim().to_lowercase();
    if g == p {
        return true;
    }
    let name_like = |s: &str| !mentions::name_words(s).is_empty();
    if name_like(gold) && name_like(pick) {
        return mentions::same_name(gold, pick);
    }
    // "my poor sister Julia" and "Your sister", "my sister's" and "my
    // sister": the same head noun on either side
    let words = |s: &str| {
        s.replace("'s", "")
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty() && !matches!(*w, "the" | "a" | "an" | "my" | "his" | "her" | "our" | "your" | "their" | "this" | "that"))
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    let (gw, pw) = (words(&g), words(&p));
    match (gw.last(), pw.last()) {
        (Some(gh), Some(ph)) => gh == ph || gw.contains(ph) || pw.contains(gh),
        _ => false,
    }
}

// The alias group a text belongs to, if any: the group's first name.
fn canonical<'a>(aliases: &'a [Vec<String>], text: &str) -> Option<&'a str> {
    let t = text.trim().to_lowercase();
    aliases.iter().find(|g| g.iter().any(|n| *n == t)).map(|g| g[0].as_str())
}

fn score(a: Score) -> Result<()> {
    let doc = parse(&a.model_dir, &a.text, a.threads)?;
    let gold = std::fs::read_to_string(&a.gold)?;
    let aliases: Vec<Vec<String>> = gold
        .lines()
        .filter_map(|l| l.strip_prefix("# alias:"))
        .map(|l| l.split(" = ").map(|n| n.trim().to_lowercase()).collect())
        .collect();
    let same = |want: &str, pick: &str| match (canonical(&aliases, want), canonical(&aliases, pick)) {
        (Some(x), Some(y)) => x == y,
        _ => matches(want, pick),
    };
    let mut by_word: std::collections::BTreeMap<String, (usize, usize)> = std::collections::BTreeMap::new();
    let (mut total, mut hit) = (0, 0);
    for line in gold.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 4 || f[3].is_empty() || f[3] == "?" {
            continue;
        }
        let (si, wi): (usize, usize) = (f[0].parse()?, f[1].parse()?);
        let m = doc.all.iter().find(|m| m.kind == Kind::Pronoun && m.sentence == si && m.start == wi);
        let pick = m.and_then(|m| m.candidates.first()).map(|c| c.text.as_str()).unwrap_or("");
        let want = f[3];
        let alt = f.get(5).and_then(|n| n.strip_prefix("alt:"));
        let accept = |w: &str| match w {
            "NONE" | "CLAUSE" => m.is_none() || pick.is_empty(),
            _ => same(w, pick),
        };
        let ok = accept(want) || alt.is_some_and(accept);
        total += 1;
        hit += ok as usize;
        let e = by_word.entry(f[2].to_lowercase()).or_insert((0, 0));
        e.0 += 1;
        e.1 += ok as usize;
        if !ok && a.verbose {
            let words = &doc.sents[si].words;
            println!("s{si} {}: want {want:?} got {pick:?}  | {}", f[2], mentions::join_words(words));
        }
    }
    println!("pronouns {total}  correct {hit}  accuracy {:.1}", 100.0 * hit as f64 / total.max(1) as f64);
    for (w, (n, k)) in &by_word {
        println!("{w:12} {k:4}/{n:<4} {:5.1}", 100.0 * *k as f64 / *n as f64);
    }
    Ok(())
}

fn main() -> Result<()> {
    match Args::parse().cmd {
        Cmd::Draft(a) => draft(a),
        Cmd::Score(a) => score(a),
    }
}
