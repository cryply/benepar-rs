// Mentions of things in a parse tree, and a first pass at saying which
// pronoun refers to which earlier mention.
//
// This is deliberately rule based and conservative. For every pronoun it
// returns a ranked list of candidates rather than one answer, so the caller
// can keep the ambiguity or let the user decide. The rules are the usual
// ones: agreement in number, a preference for names over descriptions,
// for subjects over other positions, and for recent mentions over older
// ones; reflexives bind inside their own sentence.

use crate::tree::Node;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    // A proper name: a run of NNP/NNPS inside a noun phrase.
    Name,
    // A description headed by a common noun: "the report", "two cats".
    Nominal,
    Pronoun,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Number {
    Singular,
    Plural,
    Unknown,
}

#[derive(Clone, Debug, Serialize)]
pub struct Mention {
    pub sentence: usize,
    // word indices inside the sentence, end exclusive
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub kind: Kind,
    pub head: String,
    pub number: Number,
    pub subject: bool,
    // for pronouns: ranked antecedent candidates, best first
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    pub sentence: usize,
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub score: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gender {
    Male,
    Female,
}

// A small list of common English first names, enough to tell "he" from
// "she" in ordinary text. Unknown names are left ungendered and can still
// be chosen; the list is a hint, not a filter.
const MALE: &[&str] = &[
    "james", "john", "robert", "michael", "william", "david", "richard", "joseph", "thomas",
    "charles", "daniel", "matthew", "anthony", "mark", "paul", "steven", "andrew", "kenneth",
    "george", "brian", "edward", "kevin", "jason", "jeff", "gary", "timothy", "jose", "larry",
    "frank", "scott", "eric", "stephen", "raymond", "gregory", "samuel", "benjamin", "patrick",
    "jack", "dennis", "jerry", "alexander", "henry", "peter", "adam", "nathan", "harry", "carl",
    "arthur", "ryan", "roger", "bob", "dave", "tom", "bill", "jim", "mike", "chris", "steve",
    "alex", "sam", "max", "oliver", "noah", "liam", "ethan", "lucas", "ivan", "boris", "sergey",
];
const FEMALE: &[&str] = &[
    "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara", "susan", "jessica", "sarah",
    "karen", "nancy", "lisa", "margaret", "betty", "sandra", "ashley", "dorothy", "kimberly",
    "emily", "donna", "michelle", "carol", "amanda", "melissa", "deborah", "stephanie", "rebecca",
    "laura", "sharon", "cynthia", "kathleen", "amy", "angela", "shirley", "anna", "ruth", "emma",
    "olivia", "sophia", "isabella", "mia", "charlotte", "amelia", "alice", "eve", "grace", "hannah",
    "julia", "kate", "lucy", "maria", "natalie", "rachel", "sophie", "victoria", "zoe", "jane",
    "helen", "irene", "nina", "vera", "diana", "elena", "marina", "tanya", "anya", "clara",
];

fn gender_of_name(text: &str) -> Option<Gender> {
    let mut parts = text.split_whitespace().map(|w| w.trim_matches('.').to_lowercase());
    let first = parts.next()?;
    match first.as_str() {
        "mr" | "sir" | "lord" | "king" | "prince" => return Some(Gender::Male),
        "mrs" | "ms" | "miss" | "lady" | "queen" | "princess" => return Some(Gender::Female),
        "dr" | "prof" => {}
        _ => {
            if MALE.contains(&first.as_str()) {
                return Some(Gender::Male);
            }
            if FEMALE.contains(&first.as_str()) {
                return Some(Gender::Female);
            }
            return None;
        }
    }
    let second = parts.next()?;
    if MALE.contains(&second.as_str()) {
        Some(Gender::Male)
    } else if FEMALE.contains(&second.as_str()) {
        Some(Gender::Female)
    } else {
        None
    }
}

fn pronoun_gender(p: &str) -> Option<Gender> {
    match p {
        "he" | "him" | "his" | "himself" => Some(Gender::Male),
        "she" | "her" | "hers" | "herself" => Some(Gender::Female),
        _ => None,
    }
}

fn is_name_tag(t: &str) -> bool {
    matches!(t, "NNP" | "NNPS")
}

fn is_noun_tag(t: &str) -> bool {
    matches!(t, "NN" | "NNS" | "NNP" | "NNPS")
}

fn is_pronoun_tag(t: &str) -> bool {
    matches!(t, "PRP" | "PRP$")
}

struct Leaf<'a> {
    tag: &'a str,
    word: &'a str,
}

fn leaves<'a>(t: &'a Node, out: &mut Vec<Leaf<'a>>) {
    match t {
        Node::Leaf(tag, word) => out.push(Leaf { tag, word }),
        Node::Phrase(_, kids) => kids.iter().for_each(|k| leaves(k, out)),
    }
}

fn count_leaves(t: &Node) -> usize {
    match t {
        Node::Leaf(..) => 1,
        Node::Phrase(_, kids) => kids.iter().map(count_leaves).sum(),
    }
}

fn top_label(t: &Node) -> &str {
    t.labels().first().copied().unwrap_or("")
}

// Mentions of one sentence. `sentence` is its index in the document.
pub fn mentions(tree: &Node, sentence: usize) -> Vec<Mention> {
    let mut out = Vec::new();
    walk(tree, 0, sentence, false, &mut out);
    out.sort_by_key(|m| (m.start, std::cmp::Reverse(m.end)));
    out
}

fn walk(t: &Node, start: usize, sentence: usize, subject: bool, out: &mut Vec<Mention>) {
    let kids = match t {
        Node::Leaf(..) => return,
        Node::Phrase(_, kids) => kids,
    };
    let label = top_label(t);
    let is_np = label == "NP" || label == "WHNP" || label == "NML";
    let is_clause = matches!(label, "S" | "SINV" | "SQ" | "SBAR" | "" | "TOP");

    if is_np {
        // A base NP contains no NP of its own (PPs and clauses attached
        // below it are fine, they are not part of what is named).
        let has_np_child = kids.iter().any(|k| matches!(top_label(k), "NP" | "NML"));
        if !has_np_child {
            if let Some(m) = base_np(t, start, sentence, subject) {
                out.push(m);
            }
        }
    }

    // Lone pronouns and names do not always get an NP of their own in the
    // tree (e.g. coordinated), so look at leaves directly too.
    let mut pos = start;
    let mut i = 0;
    while i < kids.len() {
        let k = &kids[i];
        let n = count_leaves(k);
        match k {
            Node::Leaf(tag, word) if is_pronoun_tag(tag) && !is_np => {
                out.push(pronoun(word, tag, pos, sentence, subject));
            }
            Node::Leaf(tag, _) if is_name_tag(tag) && !is_np => {
                // run of names outside an NP
                let mut j = i;
                let mut words = Vec::new();
                while j < kids.len() {
                    if let Node::Leaf(tg, w) = &kids[j] {
                        if is_name_tag(tg) {
                            words.push((tg.as_str(), w.as_str()));
                            j += 1;
                            continue;
                        }
                    }
                    break;
                }
                out.push(name_mention(&words, pos, sentence, subject));
                pos += words.len();
                i = j;
                continue;
            }
            Node::Phrase(..) => {
                // the first NP directly under a clause is its subject
                let child_subject = is_clause && matches!(top_label(k), "NP") && !kids[..i].iter().any(|p| top_label(p) == "NP");
                walk(k, pos, sentence, child_subject, out);
            }
            _ => {}
        }
        pos += n;
        i += 1;
    }
}

fn name_mention(words: &[(&str, &str)], start: usize, sentence: usize, subject: bool) -> Mention {
    let plural = words.last().map_or(false, |(t, _)| *t == "NNPS");
    let text: Vec<&str> = words.iter().map(|(_, w)| *w).collect();
    Mention {
        sentence,
        start,
        end: start + words.len(),
        text: join_words(&text),
        kind: Kind::Name,
        head: text.last().copied().unwrap_or("").to_string(),
        number: if plural { Number::Plural } else { Number::Singular },
        subject,
        candidates: Vec::new(),
    }
}

fn pronoun(word: &str, tag: &str, pos: usize, sentence: usize, subject: bool) -> Mention {
    let lower = word.to_lowercase();
    let number = match lower.as_str() {
        "they" | "them" | "their" | "theirs" | "themselves" | "we" | "us" | "our" | "ours" | "ourselves" => Number::Plural,
        "it" | "its" | "itself" | "he" | "him" | "his" | "himself" | "she" | "her" | "hers" | "herself" | "i" | "me" | "my" | "mine" | "myself" => Number::Singular,
        _ => Number::Unknown,
    };
    let _ = tag;
    Mention {
        sentence,
        start: pos,
        end: pos + 1,
        text: word.to_string(),
        kind: Kind::Pronoun,
        head: lower,
        number,
        subject,
        candidates: Vec::new(),
    }
}

fn base_np(t: &Node, start: usize, sentence: usize, subject: bool) -> Option<Mention> {
    let mut lv = Vec::new();
    leaves(t, &mut lv);
    // Only the words before any PP or clause inside the NP are the name.
    let core_len = match t {
        Node::Phrase(_, kids) => {
            let mut n = 0;
            for k in kids {
                if matches!(k, Node::Phrase(..)) && matches!(top_label(k), "PP" | "SBAR" | "S" | "VP" | "RRC" | "PRN") {
                    break;
                }
                n += count_leaves(k);
            }
            n
        }
        _ => 0,
    };
    let core = &lv[..core_len.min(lv.len())];
    if core.is_empty() {
        return None;
    }
    // Strip trailing punctuation and possessive markers.
    let mut end = core.len();
    while end > 0 && matches!(core[end - 1].tag, "," | "." | ":" | "POS" | "''" | "``") {
        end -= 1;
    }
    let core = &core[..end];
    if core.is_empty() {
        return None;
    }
    if core.len() == 1 && is_pronoun_tag(core[0].tag) {
        return Some(pronoun(core[0].word, core[0].tag, start, sentence, subject));
    }
    if core.iter().all(|l| is_name_tag(l.tag)) {
        let words: Vec<(&str, &str)> = core.iter().map(|l| (l.tag, l.word)).collect();
        return Some(name_mention(&words, start, sentence, subject));
    }
    let head = core.iter().rev().find(|l| is_noun_tag(l.tag))?;
    let number = match head.tag {
        "NNS" | "NNPS" => Number::Plural,
        "NN" | "NNP" => Number::Singular,
        _ => Number::Unknown,
    };
    let text: Vec<&str> = core.iter().map(|l| l.word).collect();
    Some(Mention {
        sentence,
        start,
        end: start + core.len(),
        text: join_words(&text),
        kind: Kind::Nominal,
        head: head.word.to_lowercase(),
        number,
        subject,
        candidates: Vec::new(),
    })
}

// Fill in antecedent candidates for every pronoun, looking back over the
// mentions of this and earlier sentences (in document order).
pub fn resolve(all: &mut [Mention]) {
    let snapshot: Vec<Mention> = all.to_vec();
    // genders learned from the name list and from earlier resolutions,
    // keyed by lowercased mention text
    let mut known: Vec<(String, Gender)> = Vec::new();
    for c in &snapshot {
        if c.kind == Kind::Name {
            if let Some(g) = gender_of_name(&c.text) {
                known.push((c.text.to_lowercase(), g));
            }
        }
    }
    // what each pronoun word last referred to, per sentence: a repeated
    // "it" in one sentence usually keeps its referent
    let mut last_in_sentence: Vec<(usize, String, String)> = Vec::new();
    for (i, m) in all.iter_mut().enumerate() {
        if m.kind != Kind::Pronoun {
            continue;
        }
        let p = m.head.as_str();
        // first and second person refer to the speakers, not to the text
        if matches!(p, "i" | "me" | "my" | "mine" | "myself" | "you" | "your" | "yours" | "yourself" | "we" | "us" | "our" | "ours" | "ourselves") {
            continue;
        }
        let reflexive = p.ends_with("self") || p.ends_with("selves");
        let want = pronoun_gender(p);
        let animate = want.is_some();
        let mut cands: Vec<Candidate> = Vec::new();
        for (j, c) in snapshot.iter().enumerate() {
            if j >= i || c.kind == Kind::Pronoun {
                continue;
            }
            if reflexive && c.sentence != m.sentence {
                continue;
            }
            if !reflexive && c.sentence == m.sentence && c.start >= m.start {
                continue;
            }
            if m.number != Number::Unknown && c.number != Number::Unknown && m.number != c.number {
                continue;
            }
            let key = c.text.to_lowercase();
            let have = known.iter().find(|(k, _)| *k == key).map(|(_, g)| *g);
            let mut score = 0.0f32;
            match (animate, c.kind) {
                (true, Kind::Name) => {
                    score += 2.0;
                    match (want, have) {
                        (Some(w), Some(h)) if w == h => score += 2.0,
                        (Some(_), Some(_)) => continue,
                        _ => {}
                    }
                }
                (true, Kind::Nominal) => score += 0.5,
                // "it": a thing, not a person with a known gender
                (false, Kind::Name) => {
                    if have.is_some() {
                        continue;
                    }
                }
                (false, Kind::Nominal) => score += 1.5,
                _ => {}
            }
            if c.subject {
                score += 0.5;
            }
            let distance = (m.sentence - c.sentence) as f32;
            score -= 0.5 * distance;
            if c.sentence == m.sentence && !reflexive {
                score += 0.25;
            }
            if last_in_sentence.iter().any(|(sn, h, t)| *sn == m.sentence && h == p && *t == key) {
                score += 1.0;
            }
            cands.push(Candidate {
                sentence: c.sentence,
                start: c.start,
                end: c.end,
                text: c.text.clone(),
                score,
            });
        }
        cands.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut seen: Vec<String> = Vec::new();
        cands.retain(|c| {
            let key = c.text.to_lowercase();
            if seen.contains(&key) {
                false
            } else {
                seen.push(key);
                true
            }
        });
        cands.truncate(5);
        // remember the gender a he/she resolution implies for a name
        if let (Some(w), Some(best)) = (want, cands.first()) {
            let key = best.text.to_lowercase();
            if !known.iter().any(|(k, _)| *k == key) && snapshot.iter().any(|c| c.kind == Kind::Name && c.text.to_lowercase() == key) {
                known.push((key, w));
            }
        }
        if let Some(best) = cands.first() {
            last_in_sentence.push((m.sentence, p.to_string(), best.text.to_lowercase()));
        }
        m.candidates = cands;
    }
}

// Words joined by spaces, except around a hyphen that was split out of a
// word ("owner - controlled" is printed as "owner-controlled").
fn join_words<S: AsRef<str>>(words: &[S]) -> String {
    let words: Vec<&str> = words.iter().map(|w| w.as_ref()).collect();
    words.join(" ").replace(" - ", "-")
}

// Words of every sentence with each resolved pronoun followed by its best
// antecedent in brackets: "she [Alice] would call him [Bob]".
pub fn annotate(sentences: &[Vec<String>], all: &[Mention]) -> Vec<String> {
    let mut out = Vec::new();
    for (si, words) in sentences.iter().enumerate() {
        let mut pieces: Vec<String> = words.clone();
        for m in all.iter().filter(|m| m.sentence == si && m.kind == Kind::Pronoun) {
            if let Some(c) = m.candidates.first() {
                if c.text.to_lowercase() != m.text.to_lowercase() {
                    pieces[m.start] = format!("{} [{}]", words[m.start], c.text);
                }
            }
        }
        out.push(join_words(&pieces));
    }
    out
}

// Entities for indexing: every non-pronoun mention text, lowercased, with
// the sentences it appears in, including sentences that reach it only
// through a resolved pronoun.
pub fn entities(all: &[Mention]) -> Vec<(String, Vec<usize>)> {
    let mut map: Vec<(String, Vec<usize>)> = Vec::new();
    let mut add = |key: String, s: usize| {
        if let Some(e) = map.iter_mut().find(|(k, _)| *k == key) {
            if !e.1.contains(&s) {
                e.1.push(s);
            }
        } else {
            map.push((key, vec![s]));
        }
    };
    for m in all {
        match m.kind {
            Kind::Pronoun => {
                if let Some(c) = m.candidates.first() {
                    add(c.text.to_lowercase(), m.sentence);
                }
            }
            _ => add(m.text.to_lowercase(), m.sentence),
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::treebank::parse;

    fn doc(trees: &[&str]) -> Vec<Mention> {
        let mut all = Vec::new();
        for (i, t) in trees.iter().enumerate() {
            all.extend(mentions(&parse(t).unwrap(), i));
        }
        resolve(&mut all);
        all
    }

    #[test]
    fn finds_names_nominals_and_pronouns() {
        let m = mentions(
            &parse("(S (NP (NNP Alice)) (VP (VBD sent) (NP (NNP Bob)) (NP (DT the) (JJ urgent) (NN report)) (PP (IN on) (NP (NN safety)))) (. .))").unwrap(),
            0,
        );
        let texts: Vec<(&str, Kind)> = m.iter().map(|x| (x.text.as_str(), x.kind)).collect();
        assert_eq!(
            texts,
            vec![("Alice", Kind::Name), ("Bob", Kind::Name), ("the urgent report", Kind::Nominal), ("safety", Kind::Nominal)]
        );
        assert!(m[0].subject && !m[1].subject);
        assert_eq!(m[2].head, "report");
    }

    #[test]
    fn pronouns_get_ranked_candidates() {
        let all = doc(&[
            "(S (NP (NNP Alice)) (VP (VBD told) (NP (NNP Bob)) (SBAR (IN that) (S (NP (PRP she)) (VP (MD would) (VP (VB call) (NP (PRP him))))))) (. .))",
            "(S (NP (PRP It)) (VP (VBD was) (ADJP (JJ late))) (. .))",
        ]);
        let she = all.iter().find(|m| m.text == "she").unwrap();
        assert_eq!(she.candidates[0].text, "Alice");
        let him = all.iter().find(|m| m.text == "him").unwrap();
        assert_eq!(him.candidates[0].text, "Bob");
        assert!(!him.candidates.iter().any(|c| c.text == "Alice"));
        let it = all.iter().find(|m| m.text == "It").unwrap();
        // "it" is not a person: Alice and Bob are known names, so nothing fits
        assert!(it.candidates.is_empty());
    }

    #[test]
    fn number_agreement_and_annotation() {
        let trees = [
            "(S (NP (DT The) (NNS engineers)) (VP (VBD read) (NP (DT the) (NN report))) (. .))",
            "(S (NP (PRP They)) (VP (VBD liked) (NP (PRP it))) (. .))",
        ];
        let all = doc(&trees);
        let they = all.iter().find(|m| m.text == "They").unwrap();
        assert_eq!(they.candidates[0].text, "The engineers");
        let it = all.iter().find(|m| m.text == "it").unwrap();
        assert_eq!(it.candidates[0].text, "the report");
        let words: Vec<Vec<String>> = trees
            .iter()
            .map(|t| crate::treebank::leaves(&parse(t).unwrap()).into_iter().map(|(_, w)| w).collect())
            .collect();
        let ann = annotate(&words, &all);
        assert_eq!(ann[1], "They [The engineers] liked it [the report] .");
        let ents = entities(&all);
        let report = ents.iter().find(|(k, _)| k == "the report").unwrap();
        assert_eq!(report.1, vec![0, 1]);
    }

    #[test]
    fn unknown_names_learn_gender_from_use() {
        let all = doc(&[
            "(S (NP (NNP Zorblax)) (VP (VBD met) (NP (NNP Quendra))) (. .))",
            "(S (NP (PRP He)) (VP (VBD smiled)) (. .))",
            "(S (NP (PRP She)) (VP (VBD left)) (. .))",
        ]);
        let he = all.iter().find(|m| m.text == "He").unwrap();
        assert_eq!(he.candidates[0].text, "Zorblax");
        let she = all.iter().find(|m| m.text == "She").unwrap();
        assert_eq!(she.candidates[0].text, "Quendra");
    }

    #[test]
    fn reflexive_stays_in_sentence() {
        let all = doc(&[
            "(S (NP (NNP Carol)) (VP (VBD left)) (. .))",
            "(S (NP (NNP Dave)) (VP (VBD hurt) (NP (PRP himself))) (. .))",
        ]);
        let m = all.iter().find(|m| m.text == "himself").unwrap();
        assert_eq!(m.candidates[0].text, "Dave");
        assert_eq!(m.candidates.len(), 1);
    }
}
