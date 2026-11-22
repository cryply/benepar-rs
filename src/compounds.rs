// Hyphenated compounds ("owner-controlled", "16-year-old") found in a parse
// tree, with the role of the left part worked out from the head's tag.
//
// The tokenizer splits such words at the hyphen and the parser brackets the
// pieces as the revised treebank does: (ADJP (NN owner) (HYPH -) (VBN
// controlled)). The tree gives the parts and their tags but no roles; the
// rules here are the usual head-driven ones (Roeper and Siegel's first
// sister principle): the right element is the head, and its type fixes what
// the left element is.
//
// - N-Ving: the left noun is the object of the verb. "owner-controlling"
//   means X controls owner.
// - N-Ved: the object has become X itself, so the left noun is some other
//   role: agent (owner-controlled), instrument (hand-made) or place
//   (London-based). The tree cannot tell these apart; the role is left
//   empty for a lexicon or a model to fill in.
// - Adv-Ved, Adv-Ving: manner (well-known, fast-acting).
// - N-Adj: complement of the adjective (oil-rich, user-friendly).
// - Adj-N, Num-N: an attributive phrase (long-term, three-page).
// - Num-N-Adj: a measure phrase (16-year-old).
// - N-N: a modifier or, with three or more parts, coordination
//   (Rome-Berlin-Tokyo).
// - Anything else (state-of-the-art, easy-to-use, must-see) is a collapsed
//   phrase and reads as ordinary syntax without the hyphens.

use crate::tree::Node;
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Compound {
    // word positions, end exclusive
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub left: String,
    pub head: String,
    pub left_tag: String,
    pub head_tag: String,
    pub pattern: String,
    // role of the left part relative to the head; None when the pattern
    // does not decide it
    pub role: Option<String>,
    // paraphrase with X for the modified noun
    pub reading: Option<String>,
}

fn is_noun(tag: &str) -> bool {
    matches!(tag, "NN" | "NNS" | "NNP" | "NNPS")
}

fn leaves(n: &Node) -> Vec<(&str, &str)> {
    match n {
        Node::Leaf(t, w) => vec![(t.as_str(), w.as_str())],
        Node::Phrase(_, kids) => kids.iter().flat_map(leaves).collect(),
    }
}

fn word_count(n: &Node) -> usize {
    match n {
        Node::Leaf(..) => 1,
        Node::Phrase(_, kids) => kids.iter().map(word_count).sum(),
    }
}

// Words of a segment joined back into one string: "owner-operator".
fn join(ws: &[(&str, &str)]) -> String {
    let mut s = String::new();
    for (i, (t, w)) in ws.iter().enumerate() {
        if i > 0 && *t != "HYPH" && ws[i - 1].0 != "HYPH" {
            s.push(' ');
        }
        s.push_str(w);
    }
    s
}

fn classify(left: &[(&str, &str)], head: &[(&str, &str)]) -> (String, Option<&'static str>, Option<String>) {
    let (lt, lw) = (left.last().map_or("", |x| x.0), join(left));
    let first_left = left.first().map_or("", |x| x.0);
    let (ht, hw) = match head {
        [(t, w)] => (*t, w.to_string()),
        _ => return ("phrase".into(), None, None),
    };
    let left_n = is_noun(lt);
    let (pattern, role, reading): (&str, Option<&str>, Option<String>) = match ht {
        "VBN" if left_n => ("N-Ved", None, Some(format!("X {hw} by/with/in {lw}"))),
        "VBN" if lt.starts_with("RB") => ("Adv-Ved", Some("manner"), Some(format!("X {hw} {lw}"))),
        "VBN" if lt.starts_with("JJ") => ("Adj-Ved", None, None),
        "VBG" if left_n => ("N-Ving", Some("object"), Some(format!("X {hw} {lw}"))),
        "VBG" if lt.starts_with("RB") || lt.starts_with("JJ") => {
            ("Adv-Ving", Some("manner"), Some(format!("X {hw} {lw}")))
        }
        "JJ" | "JJR" | "JJS" if first_left == "CD" && lt == "NN" && left.len() > 1 => {
            ("Num-N-Adj", Some("measure"), Some(format!("X {lw} {hw}")))
        }
        "JJ" | "JJR" | "JJS" if left_n => ("N-Adj", Some("complement"), Some(format!("X {hw} in/of/to {lw}"))),
        "JJ" | "JJR" | "JJS" if lt.starts_with("JJ") && hw.ends_with("ed") => ("Adj-N-ed", Some("attribute"), None),
        "JJ" | "JJR" | "JJS" if lt.starts_with("JJ") => ("Adj-Adj", Some("modifier"), Some(format!("X {lw} {hw}"))),
        "JJ" | "JJR" | "JJS" if lt.starts_with("RB") => ("Adv-Adj", Some("manner"), Some(format!("X {lw} {hw}"))),
        "NN" | "NNS" | "NNP" | "NNPS" if lt.starts_with("JJ") => {
            ("Adj-N", Some("attributive"), Some(format!("X of {lw} {hw}")))
        }
        "NN" | "NNS" | "NNP" | "NNPS" if lt == "CD" => ("Num-N", Some("attributive"), Some(format!("X of {lw} {hw}"))),
        "NN" | "NNS" | "NNP" | "NNPS" if left_n => ("N-N", Some("modifier"), None),
        _ => ("phrase", None, None),
    };
    (pattern.to_string(), role, reading)
}

// A run of kids "A - B - C" starting at kid i: the kids and the number of
// kids it covers.
fn run_len(kids: &[Node], i: usize) -> usize {
    let is_hyph = |n: &Node| matches!(n, Node::Leaf(t, _) if t == "HYPH");
    let mut n = 1;
    while i + n + 1 < kids.len() && is_hyph(&kids[i + n]) && !is_hyph(&kids[i + n + 1]) {
        n += 2;
    }
    n
}

fn collect(node: &Node, start: usize, out: &mut Vec<Compound>) {
    let kids = match node {
        Node::Leaf(..) => return,
        Node::Phrase(_, kids) => kids,
    };
    let mut pos = start;
    let mut i = 0;
    while i < kids.len() {
        let n = run_len(kids, i);
        if n == 1 {
            collect(&kids[i], pos, out);
            pos += word_count(&kids[i]);
            i += 1;
            continue;
        }
        let run = &kids[i..i + n];
        let words: Vec<(&str, &str)> = run.iter().flat_map(leaves).collect();
        let segs: Vec<&Node> = run.iter().step_by(2).collect();
        let (left_words, head_words): (Vec<(&str, &str)>, Vec<(&str, &str)>) = if segs.len() == 2 {
            (leaves(segs[0]), leaves(segs[1]))
        } else {
            (
                run[..n - 2].iter().flat_map(leaves).collect(),
                leaves(segs[segs.len() - 1]),
            )
        };
        let all_nouns = segs.iter().all(|s| matches!(s, Node::Leaf(t, _) if is_noun(t)));
        let (pattern, role, reading) = if segs.len() > 2 && all_nouns {
            ("N-N".to_string(), Some("coordinate"), None)
        } else if segs.len() > 2 {
            ("phrase".to_string(), None, None)
        } else {
            classify(&left_words, &head_words)
        };
        let reading = if pattern == "phrase" {
            Some(words.iter().filter(|(t, _)| *t != "HYPH").map(|(_, w)| *w).collect::<Vec<_>>().join(" "))
        } else {
            reading
        };
        let len: usize = run.iter().map(word_count).sum();
        out.push(Compound {
            start: pos,
            end: pos + len,
            text: join(&words),
            left: join(&left_words),
            head: join(&head_words),
            left_tag: left_words.last().map_or("", |x| x.0).to_string(),
            head_tag: if head_words.len() == 1 { head_words[0].0.to_string() } else { "phrase".into() },
            pattern,
            role: role.map(str::to_string),
            reading,
        });
        pos += len;
        i += n;
    }
}

/// Hyphenated compounds in a tree, in word order. Nested compounds
/// ("owner-operator-controlled") are reported once, as the outer one.
pub fn find(tree: &Node) -> Vec<Compound> {
    let mut out = Vec::new();
    collect(tree, 0, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::treebank;

    fn one(s: &str) -> Compound {
        let v = find(&treebank::parse(s).unwrap());
        assert_eq!(v.len(), 1, "{s}");
        v.into_iter().next().unwrap()
    }

    #[test]
    fn head_decides_the_pattern() {
        let c = one("(NP (DT The) (ADJP (NN owner) (HYPH -) (VBN controlled)) (NNS firms))");
        assert_eq!((c.start, c.end, c.text.as_str()), (1, 4, "owner-controlled"));
        assert_eq!((c.pattern.as_str(), c.role, c.left.as_str(), c.head.as_str()), ("N-Ved", None, "owner", "controlled"));
        assert_eq!(c.reading.as_deref(), Some("X controlled by/with/in owner"));

        let c = one("(NP (DT an) (ADJP (NN owner) (HYPH -) (VBG controlling)) (NN stake))");
        assert_eq!((c.pattern.as_str(), c.role.as_deref()), ("N-Ving", Some("object")));
        assert_eq!(c.reading.as_deref(), Some("X controlling owner"));

        let c = one("(NP (ADJP (RB well) (HYPH -) (VBN known)) (NN fact))");
        assert_eq!((c.pattern.as_str(), c.role.as_deref()), ("Adv-Ved", Some("manner")));

        let c = one("(NP (ADJP (NN oil) (HYPH -) (JJ rich)) (NN state))");
        assert_eq!((c.pattern.as_str(), c.role.as_deref()), ("N-Adj", Some("complement")));

        let c = one("(NP (NML (JJ long) (HYPH -) (NN term)) (NN plan))");
        assert_eq!((c.pattern.as_str(), c.role.as_deref()), ("Adj-N", Some("attributive")));
        assert_eq!(c.reading.as_deref(), Some("X of long term"));
    }

    #[test]
    fn chains_and_phrases() {
        let c = one("(NP (DT The) (ADJP (NP (NN owner) (HYPH -) (NN operator)) (HYPH -) (VBN controlled)) (NNS trucks))");
        assert_eq!((c.text.as_str(), c.left.as_str(), c.pattern.as_str()), ("owner-operator-controlled", "owner-operator", "N-Ved"));
        assert_eq!((c.start, c.end), (1, 6));

        let c = one("(NP (ADJP (NP (CD 16) (HYPH -) (NN year)) (HYPH -) (JJ old)) (NN girl))");
        assert_eq!((c.pattern.as_str(), c.role.as_deref()), ("Num-N-Adj", Some("measure")));
        assert_eq!(c.reading.as_deref(), Some("X 16-year old"));

        let c = one("(NP (ADJP (NP (NNP New) (NNP York)) (HYPH -) (VBN based)) (NN company))");
        assert_eq!((c.text.as_str(), c.pattern.as_str()), ("New York-based", "N-Ved"));

        let c = one("(NP (NML (NNP Rome) (HYPH -) (NNP Berlin) (HYPH -) (NNP Tokyo)) (NN axis))");
        assert_eq!((c.pattern.as_str(), c.role.as_deref()), ("N-N", Some("coordinate")));

        let c = one("(NP (DT a) (NML (NML (NN state)) (HYPH -) (PP (IN of) (HYPH -) (NP (DT the) (HYPH -) (NN art)))) (NN system))");
        assert_eq!((c.text.as_str(), c.pattern.as_str(), c.role), ("state-of-the-art", "phrase".into(), None));
        assert_eq!(c.reading.as_deref(), Some("state of the art"));
    }

    #[test]
    fn flat_run_inside_a_bigger_phrase() {
        let v = find(&treebank::parse("(NP (DT a) (JJ big) (NN truck) (HYPH -) (NN driver) (. .))").unwrap());
        assert_eq!(v.len(), 1);
        assert_eq!((v[0].text.as_str(), v[0].start, v[0].end, v[0].pattern.as_str()), ("truck-driver", 2, 5, "N-N"));
    }
}
