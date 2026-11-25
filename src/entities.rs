// The entity index of a text: who and what is mentioned, where, what is
// said about each one, and which events it takes part in. Plain data for
// whoever keeps a memory of texts; nothing here depends on any store.
//
// Entities are the groups of mentions that --pronominalize uses: a name
// and its other forms ("Alice Smith", "Smith"), a definite description
// and its repeats ("the new engine", "the engine"). Pronouns join the
// entity of their best antecedent. Descriptors are what the text says
// about an entity: words inside its mentions ("the red-haired man":
// red-haired, man), titles ("Governor Angeloz": Governor), appositions
// ("Smith, the manager,": the manager) and complements of linking verbs
// ("Smith became president": president). An implicit mention later ("the
// red-haired one", "the governor") can be looked up through them.

use crate::mentions::{self, Gender, Kind, Mention, Number};
use crate::pronominalize::{self, mention_at};
use crate::skeleton::{self, Event};
use crate::tree::Node;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Occurrence {
    pub sentence: usize,
    pub start: usize,
    pub end: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<(usize, usize)>,
    pub text: String,
    pub kind: Kind,
    // for pronouns: how far the chosen antecedent was ahead of the next
    #[serde(skip_serializing_if = "Option::is_none")]
    pub margin: Option<f32>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Descriptor {
    pub text: String,
    // modifier, title, apposition, complement
    pub source: String,
    pub sentence: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Role {
    pub sentence: usize,
    // index into the events of that sentence
    pub event: usize,
    pub verb: String,
    pub role: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Entity {
    pub id: usize,
    // the first mention, which is usually the fullest
    pub name: String,
    pub kind: Kind,
    pub number: Number,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gender: Option<Gender>,
    pub mentions: Vec<Occurrence>,
    pub descriptors: Vec<Descriptor>,
    pub roles: Vec<Role>,
}

const TITLES: &[&str] = &[
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor", "sir", "lord", "lady", "dame", "president", "governor", "senator", "judge",
    "justice", "mayor", "general", "colonel", "major", "captain", "lieutenant", "sergeant", "admiral", "king", "queen", "prince",
    "princess", "duke", "pope", "bishop", "father", "reverend", "rev", "rabbi", "imam", "chief", "director", "chairman", "chairwoman",
    "ceo", "coach", "doctor", "secretary", "minister", "chancellor", "ambassador", "sheriff", "officer", "detective", "inspector",
    "representative", "rep", "congressman", "congresswoman", "speaker", "emperor", "saint", "st",
];

const STOP_MODIFIERS: &[&str] = &["the", "a", "an", "this", "that", "these", "those", "his", "her", "its", "their", "my", "our", "your", "'s", "s"];

// Words of a mention that describe it: everything before the head except
// determiners, plus the head of a description; titles before a name.
fn own_descriptors(m: &Mention, tags: &[String], words: &[String], out: &mut Vec<Descriptor>) {
    let mut head_at = m.end - 1;
    for i in (m.start..m.end).rev() {
        if words[i] == m.head {
            head_at = i;
            break;
        }
    }
    let mut mods = Vec::new();
    for i in m.start..head_at {
        let w = words[i].to_lowercase();
        let t = tags[i].as_str();
        if STOP_MODIFIERS.contains(&w.as_str()) || t == "POS" || t == "DT" || t == "PRP$" || t == "CC" || t == "," {
            continue;
        }
        if m.kind == Kind::Name {
            if TITLES.contains(&w.trim_end_matches('.')) {
                out.push(Descriptor { text: words[i].clone(), source: "title".to_string(), sentence: m.sentence });
            }
            continue;
        }
        if t.starts_with("JJ") || t.starts_with("NN") || t == "VBN" || t == "VBG" || t == "CD" || t == "HYPH" {
            mods.push(words[i].clone());
        }
    }
    if m.kind == Kind::Nominal {
        // a hyphenated run stays one word: "red - haired"
        let text = mentions::join_words(&mods);
        for part in text.split_whitespace() {
            out.push(Descriptor { text: part.to_string(), source: "modifier".to_string(), sentence: m.sentence });
        }
        out.push(Descriptor { text: m.head.clone(), source: "head".to_string(), sentence: m.sentence });
    }
}

fn label(t: &Node) -> &str {
    match t {
        Node::Phrase(l, _) => l.as_str(),
        Node::Leaf(..) => "",
    }
}

fn size(t: &Node) -> usize {
    match t {
        Node::Leaf(..) => 1,
        Node::Phrase(_, k) => k.iter().map(size).sum(),
    }
}

type Span = (usize, usize);

// Appositions in a tree: pairs of NP spans separated by a comma under one
// NP ("Smith , the manager ,").
fn appositions(t: &Node, start: usize, out: &mut Vec<(Span, Span)>) {
    let Node::Phrase(l, kids) = t else { return };
    let mut pos = start;
    let mut spans = Vec::new();
    for k in kids {
        spans.push((pos, pos + size(k), label(k).to_string(), matches!(k, Node::Leaf(t, _) if t == ",")));
        appositions(k, pos, out);
        pos += size(k);
    }
    if l == "NP" && spans.len() >= 3 {
        for w in spans.windows(3) {
            if w[0].2 == "NP" && w[1].3 && w[2].2 == "NP" {
                out.push(((w[0].0, w[0].1), (w[2].0, w[2].1)));
            }
        }
    }
}

/// The entity index of parsed sentences. `all` are the resolved mentions
/// of the whole text (with spans attached when available), `events` the
/// skeleton of every sentence.
pub fn entities(trees: &[Node], all: &[Mention], events: &[Vec<Event>]) -> Vec<Entity> {
    let mut groups = Vec::new();
    let mut of = pronominalize::group(all, &mut groups);
    pronominalize::learn_genders(all, &of, &mut groups);
    link_titles(all, &mut of, &groups);
    // pronouns join their best antecedent's entity
    for (i, m) in all.iter().enumerate() {
        if m.kind != Kind::Pronoun {
            continue;
        }
        let Some(c) = m.candidates.first() else { continue };
        of[i] = mention_at(all, c.sentence, c.start, c.end).and_then(|j| of[j]);
    }
    let expanded: Vec<Node> = trees.iter().map(skeleton::expand).collect();
    let tagged: Vec<(Vec<String>, Vec<String>)> = expanded
        .iter()
        .map(|t| {
            let leaves = crate::treebank::leaves(t);
            (leaves.iter().map(|(t, _)| t.clone()).collect(), leaves.iter().map(|(_, w)| w.clone()).collect())
        })
        .collect();
    let mut out: Vec<Entity> = groups
        .iter()
        .enumerate()
        .map(|(id, g)| Entity {
            id,
            name: g.first.clone(),
            kind: if g.key.is_some() { Kind::Nominal } else { Kind::Name },
            number: g.number,
            gender: g.gender,
            mentions: Vec::new(),
            descriptors: Vec::new(),
            roles: Vec::new(),
        })
        .collect();
    for (i, m) in all.iter().enumerate() {
        let Some(e) = of[i] else { continue };
        out[e].mentions.push(Occurrence {
            sentence: m.sentence,
            start: m.start,
            end: m.end,
            span: m.span,
            text: m.text.clone(),
            kind: m.kind,
            margin: if m.kind == Kind::Pronoun { mentions::margin(m) } else { None },
        });
        if m.kind != Kind::Pronoun {
            let (tags, words) = &tagged[m.sentence];
            own_descriptors(m, tags, words, &mut out[e].descriptors);
        }
    }
    // appositions
    for (si, t) in expanded.iter().enumerate() {
        let mut pairs = Vec::new();
        appositions(t, 0, &mut pairs);
        for (a, b) in pairs {
            for (x, y) in [(a, b), (b, a)] {
                let Some(ex) = all.iter().position(|m| m.sentence == si && (m.start, m.end) == x && m.kind != Kind::Pronoun).and_then(|i| of[i]) else { continue };
                let Some(my) = all.iter().find(|m| m.sentence == si && m.start == y.0 && m.end <= y.1 && m.kind != Kind::Pronoun) else { continue };
                if my.kind == Kind::Nominal {
                    out[ex].descriptors.push(Descriptor { text: my.text.clone(), source: "apposition".to_string(), sentence: si });
                }
            }
        }
    }
    // roles in events, and complements as descriptors
    for (si, evs) in events.iter().enumerate() {
        for (ei, ev) in evs.iter().enumerate() {
            let subject = ev.args.iter().find(|a| a.role == "subject").and_then(|a| a.mention).and_then(|mi| of.get(index_in_all(all, si, mi)?).copied().flatten());
            for a in &ev.args {
                let Some(mi) = a.mention else { continue };
                let Some(e) = index_in_all(all, si, mi).and_then(|i| of[i]) else { continue };
                out[e].roles.push(Role { sentence: si, event: ei, verb: ev.word.clone(), role: a.role.clone() });
            }
            if let (Some(e), Some(c)) = (subject, ev.args.iter().find(|a| a.role == "complement")) {
                out[e].descriptors.push(Descriptor { text: c.text.clone(), source: "complement".to_string(), sentence: si });
            }
        }
    }
    for e in out.iter_mut() {
        // the same word said twice is one descriptor; the first place stays
        let mut seen: Vec<(String, String)> = Vec::new();
        e.descriptors.retain(|d| {
            let key = (d.text.to_lowercase(), d.source.clone());
            if seen.contains(&key) {
                false
            } else {
                seen.push(key);
                true
            }
        });
    }
    out.retain(|e| !e.mentions.is_empty());
    out
}

// "the Doctor" is the one person with the title Dr., "the Governor" the
// one with Governor: definite descriptions whose head is a title word join
// that name's entity when exactly one name carries the title.
fn link_titles(all: &[Mention], of: &mut [Option<usize>], groups: &[pronominalize::Entity]) {
    let title_key = |w: &str| -> String {
        let w = w.trim_end_matches('.').to_lowercase();
        match w.as_str() {
            "dr" | "doctor" => "doctor".to_string(),
            "prof" | "professor" => "professor".to_string(),
            "rev" | "reverend" => "reverend".to_string(),
            "rep" | "representative" => "representative".to_string(),
            _ => w,
        }
    };
    // title -> entities whose first mention carries it
    let mut carriers: Vec<(String, usize)> = Vec::new();
    for (e, g) in groups.iter().enumerate() {
        // a bare "Doctor" in address is not a name carrying a title
        if g.key.is_some() || g.first.split_whitespace().count() < 2 {
            continue;
        }
        for w in g.first.split_whitespace() {
            let k = title_key(w);
            if TITLES.contains(&k.as_str()) || k == "doctor" || k == "professor" {
                carriers.push((k, e));
            }
        }
    }
    for (i, m) in all.iter().enumerate() {
        if m.kind != Kind::Nominal || m.number != Number::Singular {
            continue;
        }
        let lower = m.text.to_lowercase();
        if !lower.starts_with("the ") || lower.split_whitespace().count() != 2 {
            continue;
        }
        let k = title_key(&m.head);
        let mut hits = carriers.iter().filter(|(t, _)| *t == k).map(|(_, e)| *e).collect::<Vec<_>>();
        hits.dedup();
        if hits.len() == 1 {
            of[i] = Some(hits[0]);
        }
    }
}

// Mentions of one sentence are numbered from 0 in the skeleton; `all`
// holds every sentence's mentions in order.
fn index_in_all(all: &[Mention], sentence: usize, within: usize) -> Option<usize> {
    let first = all.iter().position(|m| m.sentence == sentence)?;
    let i = first + within;
    (i < all.len() && all[i].sentence == sentence).then_some(i)
}

/// One line per entity for the terminal.
pub fn lines(entities: &[Entity]) -> Vec<String> {
    entities
        .iter()
        .map(|e| {
            let mut s = format!("#{} {}", e.id, e.name);
            let forms: Vec<&str> = e.mentions.iter().map(|m| m.text.as_str()).filter(|t| *t != e.name).collect();
            if !forms.is_empty() {
                s.push_str(&format!(" ({})", forms.join(", ")));
            }
            if !e.descriptors.is_empty() {
                let d: Vec<String> = e.descriptors.iter().map(|d| format!("{}:{}", d.source, d.text)).collect();
                s.push_str(&format!("  [{}]", d.join("; ")));
            }
            if !e.roles.is_empty() {
                let r: Vec<String> = e.roles.iter().map(|r| format!("{} of {}", r.role, r.verb)).collect();
                s.push_str(&format!("  {{{}}}", r.join(", ")));
            }
            s
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::treebank::parse;

    fn run(trees: &[&str]) -> Vec<Entity> {
        let trees: Vec<Node> = trees.iter().map(|t| parse(t).unwrap()).collect();
        let mut all = Vec::new();
        let mut events = Vec::new();
        for (i, t) in trees.iter().enumerate() {
            let m = mentions::mentions(t, i);
            events.push(skeleton::events(t, i, &m));
            all.extend(m);
        }
        mentions::resolve(&mut all);
        entities(&trees, &all, &events)
    }

    #[test]
    fn names_descriptions_and_pronouns_group() {
        let e = run(&[
            "(S (NP (NNP Governor) (NNP Angeloz)) (VP (VBD met) (NP (DT the) (JJ red-haired) (NN engineer))) (. .))",
            "(S (NP (NNP Angeloz)) (VP (VBD became) (NP (NN president))) (. .))",
            "(S (NP (PRP He)) (VP (VBD thanked) (NP (DT the) (NN engineer))) (. .))",
        ]);
        let l = lines(&e);
        assert_eq!(l[0], "#0 Governor Angeloz (Angeloz, He)  [title:Governor; complement:president]  {subject of met, subject of became, subject of thanked}");
        assert_eq!(l[1], "#1 the red-haired engineer (the engineer)  [modifier:red-haired; head:engineer]  {object of met, object of thanked}");
        assert_eq!(e[0].mentions[2].kind, Kind::Pronoun);
        assert!(e[0].mentions[2].margin.is_some());
    }

    #[test]
    fn the_doctor_is_the_one_with_the_title() {
        let e = run(&[
            "(S (NP (NNP Dr.) (NNP Roylott)) (VP (VBD kept) (NP (DT a) (NN cheetah))) (. .))",
            "(S (NP (DT The) (NN doctor)) (VP (VBD smiled)) (. .))",
        ]);
        let r = e.iter().find(|x| x.name == "Dr. Roylott").unwrap();
        assert!(r.mentions.iter().any(|m| m.text == "The doctor"));
        assert!(r.roles.iter().any(|x| x.verb == "smiled"));
    }

    #[test]
    fn appositions_describe_both_ways() {
        let e = run(&["(S (NP (NP (NNP Smith)) (, ,) (NP (DT the) (NN manager)) (, ,)) (VP (VBD left)) (. .))"]);
        let smith = e.iter().find(|x| x.name == "Smith").unwrap();
        assert!(smith.descriptors.iter().any(|d| d.source == "apposition" && d.text == "the manager"));
        assert_eq!(smith.roles.len(), 1);
    }
}
