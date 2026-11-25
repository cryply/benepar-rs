// Scoring the pronoun rules on GAP (Webster et al., "Mind the GAP", TACL
// 2018): Wikipedia snippets, each with one pronoun and two names A and B,
// and gold labels saying whether the pronoun refers to A, to B or to
// neither. Data: https://github.com/google-research-datasets/gap-coreference
//
// For every example the snippet is parsed and resolved as usual, the
// pronoun is found by its offset, and its candidates are matched to A and
// B by span or by a shared name word ("Smith" matches "Alice Smith").
// Two readings of the candidate list are scored:
// - free: only the best candidate counts; if it is neither A nor B, the
//   prediction is "neither"
// - restricted: the best candidate that is A or B, as in the GAP setup
//   where the two names are given.

use crate::mentions::{self, Kind, Mention};
use anyhow::{Context, Result, bail};

#[derive(Clone, Debug)]
pub struct Name {
    pub text: String,
    // byte offset in the example text
    pub offset: usize,
    pub coref: bool,
}

#[derive(Clone, Debug)]
pub struct Example {
    pub id: String,
    pub text: String,
    pub pronoun: String,
    // byte offset in the example text
    pub pronoun_offset: usize,
    pub a: Name,
    pub b: Name,
}

// GAP offsets count characters; spans here are bytes.
fn char_to_byte(text: &str, n: usize) -> Option<usize> {
    if n == text.chars().count() {
        return Some(text.len());
    }
    text.char_indices().nth(n).map(|(i, _)| i)
}

fn flag(s: &str) -> Result<bool> {
    match s {
        "TRUE" | "True" | "true" => Ok(true),
        "FALSE" | "False" | "false" => Ok(false),
        _ => bail!("not a boolean: {s:?}"),
    }
}

/// Examples of a GAP .tsv file (header line first).
pub fn parse_tsv(data: &str) -> Result<Vec<Example>> {
    let mut out = Vec::new();
    for (n, line) in data.lines().enumerate().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 10 {
            bail!("line {}: {} fields, expected at least 10", n + 1, f.len());
        }
        let text = f[1].to_string();
        let off = |s: &str| -> Result<usize> {
            let c: usize = s.parse().with_context(|| format!("line {}: bad offset {s:?}", n + 1))?;
            char_to_byte(&text, c).with_context(|| format!("line {}: offset {c} past the text", n + 1))
        };
        let ex = Example {
            id: f[0].to_string(),
            pronoun: f[2].to_string(),
            pronoun_offset: off(f[3])?,
            a: Name { text: f[4].to_string(), offset: off(f[5])?, coref: flag(f[6])? },
            b: Name { text: f[7].to_string(), offset: off(f[8])?, coref: flag(f[9])? },
            text,
        };
        out.push(ex);
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    A,
    B,
}

fn overlaps(span: Option<(usize, usize)>, name: &Name) -> bool {
    match span {
        Some((s, e)) => s < name.offset + name.text.len() && name.offset < e,
        None => false,
    }
}

/// Which of the two names a candidate stands for, if either. A span
/// overlap decides first, then a shared name word; a candidate that
/// matches both names the same way is left undecided.
pub fn slot_of(ex: &Example, text: &str, span: Option<(usize, usize)>) -> Option<Slot> {
    let pick = |a: bool, b: bool| match (a, b) {
        (true, false) => Some(Some(Slot::A)),
        (false, true) => Some(Some(Slot::B)),
        (true, true) => Some(None),
        (false, false) => None,
    };
    if let Some(r) = pick(overlaps(span, &ex.a), overlaps(span, &ex.b)) {
        return r;
    }
    pick(mentions::same_name(text, &ex.a.text), mentions::same_name(text, &ex.b.text)).flatten()
}

#[derive(Clone, Debug, Default)]
pub struct Prediction {
    // the pronoun was found among the mentions at its offset
    pub found: bool,
    pub free: Option<Slot>,
    pub restricted: Option<Slot>,
    pub margin: Option<f32>,
    // sentences between the pronoun and its best candidate
    pub distance: Option<usize>,
    // how many candidates passed the hard filters
    pub candidates: usize,
}

/// Prediction for one example from its resolved mentions (with spans).
pub fn predict(ex: &Example, all: &[Mention]) -> Prediction {
    let Some(p) = all
        .iter()
        .find(|m| m.kind == Kind::Pronoun && m.span.is_some_and(|(s, _)| s == ex.pronoun_offset))
    else {
        return Prediction::default();
    };
    let free = p.candidates.first().and_then(|c| slot_of(ex, &c.text, c.span));
    let restricted = p.candidates.iter().find_map(|c| slot_of(ex, &c.text, c.span));
    let distance = p.candidates.first().map(|c| p.sentence - c.sentence);
    Prediction { found: true, free, restricted, margin: mentions::margin(p), distance, candidates: p.candidates.len() }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
}

impl Counts {
    // GAP scores the two names as separate binary decisions.
    fn add(&mut self, pred: Option<Slot>, ex: &Example) {
        for (slot, gold) in [(Slot::A, ex.a.coref), (Slot::B, ex.b.coref)] {
            match (pred == Some(slot), gold) {
                (true, true) => self.tp += 1,
                (true, false) => self.fp += 1,
                (false, true) => self.fn_ += 1,
                (false, false) => {}
            }
        }
    }
    pub fn precision(&self) -> f64 {
        100.0 * self.tp as f64 / (self.tp + self.fp).max(1) as f64
    }
    pub fn recall(&self) -> f64 {
        100.0 * self.tp as f64 / (self.tp + self.fn_).max(1) as f64
    }
    pub fn f1(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gender {
    Masculine,
    Feminine,
}

pub fn gender(pronoun: &str) -> Option<Gender> {
    match pronoun.to_lowercase().as_str() {
        "he" | "him" | "his" | "himself" => Some(Gender::Masculine),
        "she" | "her" | "hers" | "herself" => Some(Gender::Feminine),
        _ => None,
    }
}

pub const THRESHOLDS: [f32; 7] = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0];

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub examples: usize,
    pub found: usize,
    // [free, restricted] x [masculine, feminine]
    pub counts: [[Counts; 2]; 2],
    // gold positive names that come after the pronoun
    pub gold_after: usize,
    pub gold_positive: usize,
    // per threshold: free predictions of A or B with margin >= t, and how
    // many of them are right
    pub kept: [usize; THRESHOLDS.len()],
    pub kept_right: [usize; THRESHOLDS.len()],
    // free predictions of A or B by distance: same sentence, previous,
    // two or more back; and how many are right
    pub by_distance: [usize; 3],
    pub by_distance_right: [usize; 3],
    // free predictions with exactly one candidate, and how many are right
    pub lone: usize,
    pub lone_right: usize,
}

impl Report {
    pub fn add(&mut self, ex: &Example, pred: &Prediction) {
        self.examples += 1;
        if pred.found {
            self.found += 1;
        }
        let g = match gender(&ex.pronoun) {
            Some(Gender::Feminine) => 1,
            _ => 0,
        };
        self.counts[0][g].add(pred.free, ex);
        self.counts[1][g].add(pred.restricted, ex);
        for name in [&ex.a, &ex.b] {
            if name.coref {
                self.gold_positive += 1;
                if name.offset > ex.pronoun_offset {
                    self.gold_after += 1;
                }
            }
        }
        if let (Some(slot), Some(m)) = (pred.free, pred.margin) {
            let right = match slot {
                Slot::A => ex.a.coref,
                Slot::B => ex.b.coref,
            };
            for (i, t) in THRESHOLDS.iter().enumerate() {
                if m >= *t {
                    self.kept[i] += 1;
                    if right {
                        self.kept_right[i] += 1;
                    }
                }
            }
            let d = pred.distance.unwrap_or(0).min(2);
            self.by_distance[d] += 1;
            if right {
                self.by_distance_right[d] += 1;
            }
            if pred.candidates == 1 {
                self.lone += 1;
                self.lone_right += right as usize;
            }
        }
    }

    /// Masculine, feminine and overall counts for one mode (0 free, 1
    /// restricted).
    pub fn mode(&self, mode: usize) -> (Counts, Counts, Counts) {
        let [m, f] = self.counts[mode];
        let o = Counts { tp: m.tp + f.tp, fp: m.fp + f.fp, fn_: m.fn_ + f.fn_ };
        (m, f, o)
    }

    pub fn print(&self) {
        println!("examples      {}", self.examples);
        println!("pronoun found {}", self.found);
        println!("mode        M F1   F F1   bias   O F1   O P    O R");
        for (mode, name) in [(0, "free      "), (1, "restricted")] {
            let (m, f, o) = self.mode(mode);
            println!(
                "{name}  {:5.1}  {:5.1}  {:5.2}  {:5.1}  {:5.1}  {:5.1}",
                m.f1(),
                f.f1(),
                f.f1() / m.f1().max(1e-9),
                o.f1(),
                o.precision(),
                o.recall()
            );
        }
        println!("free predictions of A or B by margin:");
        for (i, t) in THRESHOLDS.iter().enumerate() {
            println!(
                "  margin >= {t:4.2}  kept {:5} ({:5.1}% of examples)  precision {:5.1}",
                self.kept[i],
                100.0 * self.kept[i] as f64 / self.examples.max(1) as f64,
                100.0 * self.kept_right[i] as f64 / self.kept[i].max(1) as f64
            );
        }
        println!("free predictions of A or B by distance to the antecedent:");
        for (i, name) in ["same sentence", "previous sentence", "further back"].iter().enumerate() {
            println!(
                "  {name:18} {:5}  precision {:5.1}",
                self.by_distance[i],
                100.0 * self.by_distance_right[i] as f64 / self.by_distance[i].max(1) as f64
            );
        }
        println!(
            "free predictions with a single candidate: {}  precision {:5.1}",
            self.lone,
            100.0 * self.lone_right as f64 / self.lone.max(1) as f64
        );
        // The rules only look back, so these occurrences cannot be chosen;
        // the same name may still occur before the pronoun too.
        println!(
            "gold A/B occurrences after the pronoun: {} of {}",
            self.gold_after, self.gold_positive
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mentions::{attach_spans, mentions, resolve};
    use crate::treebank::parse;

    const TSV: &str = "ID\tText\tPronoun\tPronoun-offset\tA\tA-offset\tA-coref\tB\tB-offset\tB-coref\tURL\n\
        t-1\tAlice Smith met Bob. Smith said she was late.\tshe\t32\tAlice Smith\t0\tTRUE\tBob\t16\tFALSE\thttp://x\n";

    #[test]
    fn reads_tsv_and_scores_one_example() {
        let ex = parse_tsv(TSV).unwrap();
        assert_eq!(ex.len(), 1);
        let ex = &ex[0];
        assert_eq!(&ex.text[ex.pronoun_offset..ex.pronoun_offset + 3], "she");
        let trees = [
            "(S (NP (NNP Alice) (NNP Smith)) (VP (VBD met) (NP (NNP Bob))) (. .))",
            "(S (NP (NNP Smith)) (VP (VBD said) (SBAR (S (NP (PRP she)) (VP (VBD was) (ADJP (JJ late)))))) (. .))",
        ];
        let mut all = Vec::new();
        for (i, t) in trees.iter().enumerate() {
            all.extend(mentions(&parse(t).unwrap(), i));
        }
        resolve(&mut all);
        let offs = vec![
            vec![(0, 5), (6, 11), (12, 15), (16, 19), (19, 20)],
            vec![(21, 26), (27, 31), (32, 35), (36, 39), (40, 44), (44, 45)],
        ];
        attach_spans(&mut all, &offs);
        let p = predict(ex, &all);
        assert!(p.found);
        // the best candidate is the later "Smith", which matches A by name
        assert_eq!(p.free, Some(Slot::A));
        assert_eq!(p.restricted, Some(Slot::A));
        let mut r = Report::default();
        r.add(ex, &p);
        let (_, f, o) = r.mode(0);
        assert_eq!((f.tp, f.fp, f.fn_), (1, 0, 0));
        assert_eq!(o.f1(), 100.0);
    }

    #[test]
    fn char_offsets_become_byte_offsets() {
        let t = "M\u{fc}nchen: he";
        assert_eq!(char_to_byte(t, 9), Some(10));
        assert_eq!(&t[10..], "he");
    }
}
