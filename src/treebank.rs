// Reading Penn Treebank bracketed trees and scoring parses against them.
//
// Scoring follows EVALB with the COLLINS.prm settings used in parsing
// papers: TOP is ignored, punctuation words do not count toward span
// positions, ADVP and PRT are treated as the same label, and matching is on
// (label, start, end) of every non-terminal above the word level.

use crate::tree::Node;
use anyhow::{bail, Result};

// Parse one bracketed tree such as "(TOP (S (NP (DT The) (NN dog)) ...))".
pub fn parse(s: &str) -> Result<Node> {
    let chars: Vec<char> = s.chars().collect();
    let mut pos = 0;
    let node = parse_node(&chars, &mut pos)?;
    Ok(node)
}

fn skip_ws(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

fn read_token(chars: &[char], pos: &mut usize) -> String {
    let start = *pos;
    while *pos < chars.len() && !chars[*pos].is_whitespace() && chars[*pos] != '(' && chars[*pos] != ')' {
        *pos += 1;
    }
    chars[start..*pos].iter().collect()
}

fn parse_node(chars: &[char], pos: &mut usize) -> Result<Node> {
    skip_ws(chars, pos);
    if *pos >= chars.len() || chars[*pos] != '(' {
        bail!("expected '(' at {pos}");
    }
    *pos += 1;
    skip_ws(chars, pos);
    let label = read_token(chars, pos);
    skip_ws(chars, pos);
    if *pos < chars.len() && chars[*pos] != '(' {
        // leaf: (TAG word)
        let word = read_token(chars, pos);
        skip_ws(chars, pos);
        if *pos >= chars.len() || chars[*pos] != ')' {
            bail!("expected ')' after leaf at {pos}");
        }
        *pos += 1;
        return Ok(Node::Leaf(label, word));
    }
    let mut kids = Vec::new();
    loop {
        skip_ws(chars, pos);
        if *pos >= chars.len() {
            bail!("unexpected end of tree");
        }
        if chars[*pos] == ')' {
            *pos += 1;
            break;
        }
        kids.push(parse_node(chars, pos)?);
    }
    Ok(Node::Phrase(label, kids))
}

// Treebank escapes back to plain text.
pub fn unescape(word: &str) -> String {
    match word {
        "-LRB-" => "(".into(),
        "-RRB-" => ")".into(),
        "-LSB-" => "[".into(),
        "-RSB-" => "]".into(),
        "-LCB-" => "{".into(),
        "-RCB-" => "}".into(),
        "``" | "''" => "\"".into(),
        _ => word.replace("\\/", "/").replace("\\*", "*"),
    }
}

pub fn leaves(t: &Node) -> Vec<(String, String)> {
    let mut out = Vec::new();
    collect_leaves(t, &mut out);
    out
}

fn collect_leaves(t: &Node, out: &mut Vec<(String, String)>) {
    match t {
        Node::Leaf(tag, word) => out.push((tag.clone(), word.clone())),
        Node::Phrase(_, kids) => kids.iter().for_each(|k| collect_leaves(k, out)),
    }
}

// Guess which words had a space after them, for text that only exists in
// tokenized form. Good enough for the parser's retokenizer, which only
// needs word boundaries to line up with subword boundaries.
pub fn guess_spaces(words: &[String]) -> Vec<bool> {
    let n = words.len();
    let mut out = vec![true; n];
    let mut open_quote = false;
    for i in 0..n {
        let next = words.get(i + 1).map(String::as_str).unwrap_or("");
        let w = words[i].as_str();
        let no_space_before_next = matches!(next, "." | "," | ";" | ":" | "?" | "!" | ")" | "]" | "}" | "%" | "'" | "..." | "'s" | "'S")
            || next.starts_with("n't")
            || matches!(next, "'re" | "'ve" | "'ll" | "'d" | "'m" | "'")
            || (next == "\"" && open_quote);
        let no_space_after = matches!(w, "(" | "[" | "{" | "$" | "#") || (w == "\"" && !open_quote);
        if w == "\"" {
            open_quote = !open_quote;
        }
        out[i] = !(no_space_before_next || no_space_after);
    }
    if n > 0 {
        out[n - 1] = false;
    }
    out
}

const PUNCT_TAGS: [&str; 7] = ["''", "``", ".", ":", ",", "-LRB-", "-RRB-"];

fn canon(label: &str) -> &str {
    if label == "PRT" { "ADVP" } else { label }
}

// Labeled spans above the word level, positions counted over non-punctuation
// words only, TOP skipped. Collapsed unary labels ("S::VP") are expanded so
// every level counts, as EVALB does.
pub fn eval_spans(t: &Node) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    walk(t, &mut pos, &mut out, true);
    out
}

fn walk(t: &Node, pos: &mut usize, out: &mut Vec<(String, usize, usize)>, root: bool) {
    match t {
        Node::Leaf(tag, _) => {
            if !PUNCT_TAGS.contains(&tag.as_str()) {
                *pos += 1;
            }
        }
        Node::Phrase(label, kids) => {
            let start = *pos;
            for k in kids {
                walk(k, pos, out, false);
            }
            let end = *pos;
            if end == start {
                return;
            }
            for l in label.split("::") {
                if l.is_empty() || (root && matches!(l, "TOP" | "ROOT" | "S1")) {
                    continue;
                }
                out.push((canon(l).to_string(), start, end));
            }
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Counts {
    pub matched: usize,
    pub gold: usize,
    pub pred: usize,
    pub sentences: usize,
    pub exact: usize,
    pub tags_right: usize,
    pub tags_total: usize,
}

impl Counts {
    pub fn add(&mut self, gold: &Node, pred: &Node) {
        let mut g = eval_spans(gold);
        let mut p = eval_spans(pred);
        g.sort();
        p.sort();
        let mut m = 0;
        let (mut i, mut j) = (0, 0);
        while i < g.len() && j < p.len() {
            match g[i].cmp(&p[j]) {
                std::cmp::Ordering::Equal => {
                    m += 1;
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        self.matched += m;
        self.gold += g.len();
        self.pred += p.len();
        self.sentences += 1;
        if m == g.len() && m == p.len() {
            self.exact += 1;
        }
        for ((gt, _), (pt, _)) in leaves(gold).iter().zip(leaves(pred).iter()) {
            if !PUNCT_TAGS.contains(&gt.as_str()) {
                self.tags_total += 1;
                if gt == pt {
                    self.tags_right += 1;
                }
            }
        }
    }

    pub fn precision(&self) -> f64 {
        pct(self.matched, self.pred)
    }

    pub fn recall(&self) -> f64 {
        pct(self.matched, self.gold)
    }

    pub fn f1(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }

    pub fn exact_match(&self) -> f64 {
        pct(self.exact, self.sentences)
    }

    pub fn tag_accuracy(&self) -> f64 {
        pct(self.tags_right, self.tags_total)
    }
}

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: &str = "(TOP (S (NP (DT The) (NN dog)) (VP (VBZ barks) (ADVP (RB loudly))) (. .)))";

    #[test]
    fn parse_and_print_roundtrip() {
        let t = parse(T).unwrap();
        assert_eq!(t.to_string(), T);
        assert_eq!(leaves(&t).len(), 5);
    }

    #[test]
    fn spans_skip_punctuation_and_top() {
        let t = parse(T).unwrap();
        let mut s = eval_spans(&t);
        s.sort();
        assert_eq!(
            s,
            vec![
                ("ADVP".to_string(), 3, 4),
                ("NP".to_string(), 0, 2),
                ("S".to_string(), 0, 4),
                ("VP".to_string(), 2, 4),
            ]
        );
    }

    #[test]
    fn f1_counts() {
        let gold = parse(T).unwrap();
        let pred = parse("(TOP (S (NP (DT The) (NN dog)) (VP (VBZ barks) (PRT (RB loudly))) (. .)))").unwrap();
        let mut c = Counts::default();
        c.add(&gold, &pred);
        assert_eq!((c.matched, c.gold, c.pred), (4, 4, 4));
        assert_eq!(c.exact, 1);
        let worse = parse("(TOP (S (NP (DT The)) (NP (NN dog)) (VP (VBZ barks) (ADVP (RB loudly))) (. .)))").unwrap();
        let mut c = Counts::default();
        c.add(&gold, &worse);
        assert_eq!((c.matched, c.gold, c.pred), (3, 4, 5));
        assert!(c.f1() > 66.0 && c.f1() < 67.0);
    }

    #[test]
    fn spaces_are_guessed() {
        let w: Vec<String> = ["He", "ca", "n't", "(", "really", ")", "pay", "$", "5", "."]
            .iter().map(|s| s.to_string()).collect();
        assert_eq!(guess_spaces(&w), [true, false, true, false, false, true, true, false, false, false]);
    }
}
