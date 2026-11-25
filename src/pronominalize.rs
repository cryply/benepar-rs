// The opposite of --canonical: rewrite a text with as many pronouns as it
// can take. A repeated mention of a name or a description becomes a
// pronoun when the pronoun would still point back to it.
//
// Candidates are the second and later mentions of an entity whose previous
// mention is in the same or the previous sentence. The pronoun's form comes
// from the entity (gender of the name, number) and from the position
// (subject, possessive before 's, otherwise object). Every replacement is
// checked by resolving the rewritten text again with the rules of
// mentions.rs: it is kept only if the new pronoun's best antecedent is the
// same entity, at least `min_margin` ahead of the next one, and no other
// pronoun changes its antecedent. Replacements are tried in text order
// and each one sees the ones kept before it.

use crate::mentions::{self, Gender, Kind, Mention, Number};
use crate::tree::Node;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Replacement {
    // position of the pronoun in the rewritten text
    pub sentence: usize,
    pub index: usize,
    pub pronoun: String,
    // the words it replaces
    pub replaced: String,
    // the first mention of the entity it refers to
    pub entity: String,
}

pub struct Rewrite {
    pub trees: Vec<Node>,
    pub replacements: Vec<Replacement>,
    // for every word of every rewritten sentence, the index of the
    // original word it came from (a pronoun: the first word it replaced)
    pub origin: Vec<Vec<usize>>,
}

impl Rewrite {
    /// Words of every rewritten sentence, joined like --canonical output.
    pub fn lines(&self) -> Vec<String> {
        self.trees
            .iter()
            .map(|t| {
                let words: Vec<String> = crate::treebank::leaves(t).into_iter().map(|(_, w)| w).collect();
                mentions::join_words(&words)
            })
            .collect()
    }
}

const DETERMINERS: &[&str] = &["the", "a", "an", "this", "that", "these", "those"];

fn person_like(head: &str) -> bool {
    mentions::is_person_noun(head)
}

fn description_key(text: &str) -> String {
    let lower = text.to_lowercase();
    let mut words: Vec<&str> = lower.split_whitespace().collect();
    if words.first().is_some_and(|w| DETERMINERS.contains(w)) {
        words.remove(0);
    }
    words.join(" ")
}

fn subset(a: &[String], b: &[String]) -> bool {
    !a.is_empty() && a.iter().all(|w| b.contains(w))
}

pub(crate) struct Entity {
    pub(crate) first: String,
    pub(crate) names: Vec<Vec<String>>,
    pub(crate) key: Option<String>,
    pub(crate) number: Number,
    pub(crate) gender: Option<Gender>,
    // more than one gender was implied by the pronouns that refer to it
    pub(crate) conflict: bool,
}

// Entity of every non-pronoun mention (None for pronouns and for mentions
// that could belong to more than one earlier entity).
pub(crate) fn group(all: &[Mention], entities: &mut Vec<Entity>) -> Vec<Option<usize>> {
    let mut of = vec![None; all.len()];
    for (i, m) in all.iter().enumerate() {
        match m.kind {
            Kind::Pronoun => continue,
            Kind::Name => {
                let words = mentions::name_words(&m.text);
                let hits: Vec<usize> = (0..entities.len())
                    .filter(|&e| {
                        entities[e].number == m.number
                            && entities[e].names.iter().any(|n| subset(&words, n) || subset(n, &words))
                            && mentions::same_name(&entities[e].first, &m.text)
                    })
                    .collect();
                match hits.len() {
                    0 => {}
                    1 => {
                        entities[hits[0]].names.push(words);
                        of[i] = Some(hits[0]);
                        continue;
                    }
                    _ => continue,
                }
                entities.push(Entity {
                    first: m.text.clone(),
                    names: vec![words],
                    key: None,
                    number: m.number,
                    gender: mentions::gender_of_name(&m.text),
                    conflict: false,
                });
            }
            Kind::Nominal => {
                // "the engine" after "the new engine": same head, words a
                // subset of the earlier description's
                let key = description_key(&m.text);
                let definite = m.text.to_lowercase().starts_with("the ");
                let kw: Vec<String> = key.split_whitespace().map(str::to_string).collect();
                let hits: Vec<usize> = (0..entities.len())
                    .filter(|&e| {
                        let Some(k) = entities[e].key.as_deref() else { return false };
                        let ew: Vec<String> = k.split_whitespace().map(str::to_string).collect();
                        entities[e].number == m.number && ew.last() == kw.last() && subset(&kw, &ew)
                    })
                    .collect();
                if hits.len() == 1 && (definite || entities[hits[0]].first.to_lowercase() == m.text.to_lowercase()) {
                    of[i] = Some(hits[0]);
                    continue;
                }
                if !hits.is_empty() {
                    continue;
                }
                entities.push(Entity {
                    first: m.text.clone(),
                    names: Vec::new(),
                    key: Some(key),
                    number: m.number,
                    gender: None,
                    conflict: false,
                });
            }
        }
        of[i] = Some(entities.len() - 1);
    }
    of
}

// Index of the mention that a candidate stands for.
pub(crate) fn mention_at(all: &[Mention], sentence: usize, start: usize, end: usize) -> Option<usize> {
    all.iter().position(|m| m.sentence == sentence && m.start == start && m.end == end && m.kind != Kind::Pronoun)
}

fn form(gender: Option<Gender>, number: Number, case: Case) -> Option<&'static str> {
    let row = match (number, gender) {
        (Number::Plural, _) => ["they", "them", "their"],
        (Number::Singular, Some(Gender::Male)) => ["he", "him", "his"],
        (Number::Singular, Some(Gender::Female)) => ["she", "her", "her"],
        (Number::Singular, None) => ["it", "it", "its"],
        (Number::Unknown, _) => return None,
    };
    Some(match case {
        Case::Subject => row[0],
        Case::Object => row[1],
        Case::Possessive => row[2],
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Case {
    Subject,
    Object,
    Possessive,
}

// Copy of `t` with leaves from..to replaced by `new` and leaf `drop`
// removed; phrases left empty are removed too.
fn edit(t: &Node, pos: &mut usize, from: usize, to: usize, new: &Node, drop: Option<usize>) -> Option<Node> {
    match t {
        Node::Leaf(..) => {
            let i = *pos;
            *pos += 1;
            if i == from {
                Some(new.clone())
            } else if (i > from && i < to) || drop == Some(i) {
                None
            } else {
                Some(t.clone())
            }
        }
        Node::Phrase(label, kids) => {
            let kids: Vec<Node> = kids.iter().filter_map(|k| edit(k, pos, from, to, new, drop)).collect();
            if kids.is_empty() { None } else { Some(Node::Phrase(label.clone(), kids)) }
        }
    }
}

fn resolve_all(trees: &[Node]) -> Vec<Mention> {
    let mut all = Vec::new();
    for (i, t) in trees.iter().enumerate() {
        all.extend(mentions::mentions(t, i));
    }
    mentions::resolve(&mut all);
    all
}

// A pronoun, keyed by (sentence, original word index), with the entity of
// its best antecedent and its margin.
type Antecedent = ((usize, usize), Option<usize>, Option<f32>);

// State of the rewrite: the current trees and, per sentence, which
// original word each current word came from.
struct State {
    trees: Vec<Node>,
    origin: Vec<Vec<usize>>,
}

impl State {
    fn current(&self, sentence: usize, orig: usize) -> Option<usize> {
        self.origin[sentence].iter().position(|&o| o == orig)
    }

    // Entity of the best antecedent of every pronoun in the current text,
    // keyed by (sentence, original word index).
    fn antecedents(&self, all: &[Mention], orig: &[Mention], of: &[Option<usize>]) -> Vec<Antecedent> {
        // a candidate is never an inserted pronoun, so its first and last
        // words are original words of an original mention
        let entity = |s: usize, start: usize, end: usize| -> Option<usize> {
            let o = &self.origin[s];
            let (os, oe) = (*o.get(start)?, *o.get(end.checked_sub(1)?)?);
            of[mention_at(orig, s, os, oe + 1)?]
        };
        all.iter()
            .filter(|m| m.kind == Kind::Pronoun)
            .map(|m| {
                let e = m.candidates.first().and_then(|c| entity(c.sentence, c.start, c.end));
                ((m.sentence, self.origin[m.sentence][m.start]), e, mentions::margin(m))
            })
            .collect()
    }
}

// Genders implied by the pronouns that refer to each entity.
pub(crate) fn learn_genders(all: &[Mention], of: &[Option<usize>], entities: &mut [Entity]) {
    for m in all.iter().filter(|m| m.kind == Kind::Pronoun) {
        let (Some(g), Some(c)) = (mentions::pronoun_gender(&m.head), m.candidates.first()) else { continue };
        let Some(e) = mention_at(all, c.sentence, c.start, c.end).and_then(|i| of[i]) else { continue };
        let ent = &mut entities[e];
        match ent.gender {
            None if !ent.conflict && ent.key.is_none() => ent.gender = Some(g),
            Some(h) if h != g => {
                ent.gender = None;
                ent.conflict = true;
            }
            _ => {}
        }
    }
}

/// Rewrite the parsed sentences of one text with repeated mentions turned
/// into pronouns where that keeps the reference clear.
pub fn pronominalize(trees: &[Node], min_margin: f32) -> Rewrite {
    let orig = resolve_all(trees);
    let mut entities = Vec::new();
    let of = group(&orig, &mut entities);
    learn_genders(&orig, &of, &mut entities);
    let tags: Vec<Vec<String>> = trees.iter().map(|t| crate::treebank::leaves(t).into_iter().map(|(t, _)| t).collect()).collect();
    let words: Vec<Vec<String>> = trees.iter().map(|t| crate::treebank::leaves(t).into_iter().map(|(_, w)| w).collect()).collect();
    let mut state = State { trees: trees.to_vec(), origin: tags.iter().map(|t| (0..t.len()).collect()).collect() };
    let mut expected = state.antecedents(&orig, &orig, &of);
    let mut replacements = Vec::new();
    // sentence of the latest mention of each entity
    let mut last: Vec<Option<usize>> = vec![None; entities.len()];
    let pronoun_entity = |m: &Mention, expected: &[Antecedent]| {
        expected.iter().find(|(k, _, _)| *k == (m.sentence, m.start)).and_then(|(_, e, _)| *e)
    };

    for (i, m) in orig.iter().enumerate() {
        if m.kind == Kind::Pronoun {
            if let Some(e) = pronoun_entity(m, &expected) {
                last[e] = Some(m.sentence);
            }
            continue;
        }
        let Some(e) = of[i] else { continue };
        let prev = last[e];
        last[e] = Some(m.sentence);
        let Some(p) = prev else { continue };
        if m.sentence > p + 1 {
            continue;
        }
        let ent = &entities[e];
        if ent.key.is_some() && m.number == Number::Singular && person_like(&m.head) {
            continue;
        }
        if ent.key.is_none() && ent.gender.is_none() && ent.number != Number::Plural {
            continue;
        }
        // a conjunct of a coordinated noun phrase ("Alice and Bob") would
        // need the case of the whole phrase, and so would a name inside a
        // longer mention; coordinated clauses and verb phrases ("read the
        // report and told") are fine
        let w = &words[m.sentence];
        let t = &tags[m.sentence];
        let possessive = t.get(m.end).is_some_and(|t| t == "POS");
        // a possessor ("Bob's boss") is inside the phrase it opens, and
        // becomes "his" all the same
        let inside = orig.iter().any(|o| {
            o.kind != Kind::Pronoun
                && o.sentence == m.sentence
                && o.start <= m.start
                && o.end >= m.end
                && (o.start, o.end) != (m.start, m.end)
                && !(possessive && o.start == m.start)
        });
        let is_cc = |j: usize| w.get(j).is_some_and(|x| matches!(x.to_lowercase().as_str(), "and" | "or" | "nor"));
        let clause_follows = t.get(m.end + 1).is_some_and(|x| x.starts_with("VB") || x == "MD" || x == "PRP" || x == "DT");
        if inside || (m.start > 0 && is_cc(m.start - 1) && !m.subject) || (is_cc(m.end) && !clause_follows) {
            continue;
        }
        let case = if possessive {
            Case::Possessive
        } else if m.subject {
            Case::Subject
        } else {
            Case::Object
        };
        // an object referring to something already named in its own
        // sentence would need a reflexive or a clause analysis; skip it
        if case == Case::Object && p == m.sentence {
            continue;
        }
        let Some(word) = form(ent.gender, ent.number, case) else { continue };
        let (Some(from), Some(last_word)) = (state.current(m.sentence, m.start), state.current(m.sentence, m.end - 1)) else { continue };
        let to = last_word + 1;
        let drop = if possessive { state.current(m.sentence, m.end) } else { None };
        let mut text = word.to_string();
        if from == 0 {
            text[..1].make_ascii_uppercase();
        }
        let tag = if case == Case::Possessive { "PRP$" } else { "PRP" };
        let leaf = Node::leaf(tag, &text);
        let mut pos = 0;
        let Some(tree) = edit(&state.trees[m.sentence], &mut pos, from, to, &leaf, drop) else { continue };
        let mut trial = State { trees: state.trees.clone(), origin: state.origin.clone() };
        trial.trees[m.sentence] = tree;
        let o = &mut trial.origin[m.sentence];
        let mut j = o.len();
        while j > 0 {
            j -= 1;
            if (j > from && j < to) || drop == Some(j) {
                o.remove(j);
            }
        }
        let all = resolve_all(&trial.trees);
        let now = trial.antecedents(&all, &orig, &of);
        let new = now.iter().find(|(k, _, _)| *k == (m.sentence, m.start));
        let ok_new = new.is_some_and(|(_, ne, g)| *ne == Some(e) && g.is_some_and(|g| g >= min_margin));
        let ok_rest = expected.iter().all(|(k, ee, _)| now.iter().any(|(nk, ne, _)| nk == k && ne == ee));
        if !(ok_new && ok_rest) {
            continue;
        }
        let replaced: Vec<&str> = w[m.start..m.end].iter().map(|s| s.as_str()).collect();
        let mut replaced = mentions::join_words(&replaced);
        if possessive {
            replaced.push(' ');
            replaced.push_str(&w[m.end]);
        }
        replacements.push(Replacement {
            sentence: m.sentence,
            index: from,
            pronoun: text,
            replaced,
            entity: ent.first.clone(),
        });
        expected = now;
        state = trial;
    }
    // Replacements go left to right, so a later one never moves the index
    // of an earlier one.
    Rewrite { trees: state.trees, replacements, origin: state.origin }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::treebank::parse;

    fn run(trees: &[&str], min_margin: f32) -> Rewrite {
        let trees: Vec<Node> = trees.iter().map(|t| parse(t).unwrap()).collect();
        pronominalize(&trees, min_margin)
    }

    #[test]
    fn repeated_names_become_pronouns() {
        let r = run(
            &[
                "(S (NP (NNP Alice)) (VP (VBD sent) (NP (NNP Bob)) (NP (DT the) (NN report))) (. .))",
                "(S (NP (NNP Bob)) (VP (VBD read) (NP (DT the) (NN report))) (. .))",
                "(S (NP (NP (NNP Bob) (POS 's)) (NN boss)) (VP (VBD liked) (NP (NNP Alice))) (. .))",
            ],
            0.0,
        );
        let lines = r.lines();
        assert_eq!(lines[0], "Alice sent Bob the report .");
        assert_eq!(lines[1], "He read it .");
        // Alice was last named two sentences back, so she stays
        assert_eq!(lines[2], "His boss liked Alice .");
        let he = &r.replacements[0];
        assert_eq!((he.sentence, he.index, he.pronoun.as_str(), he.entity.as_str()), (1, 0, "He", "Bob"));
        assert_eq!(r.replacements[2].replaced, "Bob 's");
        assert_eq!(r.origin[1], vec![0, 1, 2, 4]);
        assert_eq!(r.origin[2], vec![0, 2, 3, 4, 5]);
    }

    #[test]
    fn ambiguous_or_far_mentions_stay() {
        // two women: "she" for the second Carol could also be Dana
        let r = run(
            &[
                "(S (NP (NNP Carol)) (VP (VBD met) (NP (NNP Dana))) (. .))",
                "(S (NP (NNP Dana)) (VP (VBD left)) (. .))",
            ],
            0.5,
        );
        assert_eq!(r.lines()[1], "Dana left .");
        // three sentences apart: no pronoun
        let r = run(
            &[
                "(S (NP (NNP Eve)) (VP (VBD slept)) (. .))",
                "(S (NP (DT The) (NN dog)) (VP (VBD barked)) (. .))",
                "(S (NP (DT The) (NN cat)) (VP (VBD ran)) (. .))",
                "(S (NP (NNP Eve)) (VP (VBD woke)) (. .))",
            ],
            0.0,
        );
        assert_eq!(r.lines()[3], "Eve woke .");
    }

    #[test]
    fn people_nouns_and_unknown_genders_stay() {
        let r = run(
            &[
                "(S (NP (DT The) (NN teacher)) (VP (VBD met) (NP (NNP Zorblax))) (. .))",
                "(S (NP (DT The) (NN teacher)) (VP (VBD thanked) (NP (NNP Zorblax))) (. .))",
            ],
            0.0,
        );
        assert_eq!(r.lines()[1], "The teacher thanked Zorblax .");
        assert!(r.replacements.is_empty());
    }

    #[test]
    fn conjuncts_stay_and_coordinations_become_they() {
        let r = run(
            &[
                "(S (NP (NNP John) (CC and) (NNP Mary)) (VP (VBD bought) (NP (DT a) (NN house))) (. .))",
                "(S (NP (NNP John) (CC and) (NNP Mary)) (VP (VBD painted) (NP (DT the) (NN house))) (. .))",
                "(S (NP (NNP Mary)) (VP (VBD smiled)) (. .))",
            ],
            0.0,
        );
        let lines = r.lines();
        assert_eq!(lines[1], "They painted it .");
        // Mary was only named inside "John and Mary", which the rules do
        // not offer as an antecedent of "she"
        assert_eq!(lines[2], "Mary smiled .");
    }

    #[test]
    fn other_pronouns_keep_their_antecedents() {
        // "He" for George would resolve to Frank, the earlier subject
        let r = run(
            &[
                "(S (NP (NNP Frank)) (VP (VBD called) (NP (NNP George))) (. .))",
                "(S (NP (NNP George)) (VP (VBD thanked) (NP (PRP him))) (. .))",
            ],
            0.0,
        );
        assert_eq!(r.lines()[1], "George thanked him .");
    }
}
