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
    // byte span in the original text, once attach_spans has run
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<(usize, usize)>,
    // a name inside a flat coordination ("Mary" in "John and Mary"); not
    // offered as an antecedent, the rules cannot tell when one conjunct
    // is meant
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub conjunct: bool,
    // object of a preposition of place ("in Evanston"): a name here is
    // rarely a person
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub locative: bool,
    // possessor inside a longer phrase ("Smith" in "Smith's room")
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub possessor: bool,
    // subject of a verb of speaking ("Holmes" in "Holmes answered")
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub speaker: bool,
    // after "to" or "from" ("drove to Leatherhead"): weak evidence of a
    // place, enough once it repeats
    #[serde(skip)]
    pub(crate) motion: bool,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<(usize, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Gender {
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

pub(crate) fn gender_of_name(text: &str) -> Option<Gender> {
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

pub(crate) fn pronoun_gender(p: &str) -> Option<Gender> {
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

/// Is this word a personal, possessive or reflexive pronoun.
pub fn is_pronoun_word(w: &str) -> bool {
    matches!(
        w.to_lowercase().as_str(),
        "i" | "me" | "my" | "mine" | "myself" | "you" | "your" | "yours" | "yourself" | "yourselves" | "he" | "him" | "his" | "himself"
            | "she" | "her" | "hers" | "herself" | "it" | "its" | "itself" | "we" | "us" | "our" | "ours" | "ourselves" | "they" | "them"
            | "their" | "theirs" | "themselves"
    )
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

// An NP ending in a possessive marker: "Smith 's", "the doctor 's".
fn is_possessor(t: &Node) -> bool {
    let mut lv = Vec::new();
    leaves(t, &mut lv);
    matches!(lv.last(), Some(l) if l.tag == "POS")
}

// Mentions of one sentence. `sentence` is its index in the document.
pub fn mentions(tree: &Node, sentence: usize) -> Vec<Mention> {
    let mut out = Vec::new();
    walk(tree, 0, sentence, false, false, false, false, &mut out);
    out.sort_by_key(|m| (m.start, std::cmp::Reverse(m.end)));
    out
}

const PLACE_PREPOSITIONS: &[&str] = &[
    "in", "at", "near", "into", "onto", "across", "through", "within", "outside", "inside", "around", "throughout", "toward", "towards",
];

const SPEECH_VERBS: &[&str] = &[
    "said", "says", "say", "answered", "replied", "asked", "cried", "remarked", "observed", "continued", "added", "exclaimed",
    "murmured", "whispered", "shouted", "returned", "explained", "inquired", "demanded", "interrupted", "muttered", "declared",
    "announced", "told", "began", "went on", "put in", "suggested", "repeated", "called",
];

// Does the clause's verb phrase start with a verb of speaking.
fn speech_clause(kids: &[Node]) -> bool {
    kids.iter().any(|k| {
        top_label(k) == "VP"
            && matches!(k, Node::Phrase(_, vk) if matches!(vk.first(), Some(Node::Leaf(t, w)) if t.starts_with("VB") && SPEECH_VERBS.contains(&w.to_lowercase().as_str())))
    })
}

// A clause complement right in the clause, in its verb phrase or in its
// adjective phrase: the S in "it was his custom to smoke", the SBAR in
// "it is my belief that..." and in "made it impossible for me to tell".
fn has_clause_complement(kids: &[Node]) -> bool {
    kids.iter().any(|k| {
        matches!(top_label(k), "S" | "SBAR")
            || (matches!(top_label(k), "VP" | "ADJP") && matches!(k, Node::Phrase(_, vk) if has_clause_complement(vk)))
    })
}

// "It is my belief that...", "it became clear to me that...": the subject
// "it" stands for the clause that follows, not for anything before it.
fn expletive_it(kids: &[Node]) -> bool {
    let subject = kids.iter().find(|k| top_label(k) == "NP");
    let is_it = matches!(subject, Some(Node::Phrase(_, nk)) if nk.len() == 1 && matches!(&nk[0], Node::Leaf(t, w) if t == "PRP" && w.eq_ignore_ascii_case("it")));
    is_it && has_clause_complement(kids)
}

fn walk(t: &Node, start: usize, sentence: usize, subject: bool, locative: bool, motion: bool, speaker: bool, out: &mut Vec<Mention>) {
    let kids = match t {
        Node::Leaf(..) => return,
        Node::Phrase(_, kids) => kids,
    };
    let label = top_label(t);
    let is_np = label == "NP" || label == "WHNP" || label == "NML";
    let is_clause = matches!(label, "S" | "SINV" | "SQ" | "SBAR" | "" | "TOP");
    let is_speech = is_clause && speech_clause(kids);
    let skip_it = is_clause && expletive_it(kids);
    // the NP right after a preposition of place
    let place_pp = label == "PP"
        && matches!(kids.first(), Some(Node::Leaf(t, w)) if (t == "IN" || t == "TO") && PLACE_PREPOSITIONS.contains(&w.to_lowercase().as_str()));
    let motion_pp = label == "PP" && matches!(kids.first(), Some(Node::Leaf(t, w)) if (t == "IN" || t == "TO") && matches!(w.to_lowercase().as_str(), "to" | "from"));

    if is_np {
        // A base NP contains no NP of its own (PPs and clauses attached
        // below it are fine, they are not part of what is named). A
        // possessor NP ("Smith's chamber") does not count: the whole
        // phrase is what is named, and the possessor is a mention of its
        // own below.
        let has_np_child = kids.iter().any(|k| matches!(top_label(k), "NP" | "NML") && !is_possessor(k));
        if !has_np_child {
            if let Some(mut m) = base_np(t, start, sentence, subject) {
                m.locative = locative;
                m.motion = motion;
                m.possessor = is_possessor(t);
                m.speaker = speaker;
                out.push(m);
            }
        }
    }

    // Lone pronouns and names do not always get an NP of their own in the
    // tree (e.g. coordinated), so look at leaves directly too. In a flat
    // coordination ("John and Mary") the names are mentions of their own.
    let coordination = is_np && kids.iter().any(|k| matches!(k, Node::Leaf(t, _) if t == "CC"));
    let mut pos = start;
    let mut i = 0;
    while i < kids.len() {
        let k = &kids[i];
        let n = count_leaves(k);
        match k {
            // a possessive inside a longer NP ("his support") is a mention
            // of its own as well
            Node::Leaf(tag, word) if is_pronoun_tag(tag) && (!is_np || (tag == "PRP$" && kids.len() > 1)) => {
                out.push(pronoun(word, tag, pos, sentence, subject));
            }
            Node::Leaf(tag, _) if is_name_tag(tag) && (!is_np || coordination) => {
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
                let mut m = name_mention(&words, pos, sentence, subject && !coordination);
                m.conjunct = coordination;
                m.locative = locative;
                out.push(m);
                pos += words.len();
                i = j;
                continue;
            }
            Node::Phrase(..) => {
                // the first NP directly under a clause is its subject
                let child_subject = is_clause && matches!(top_label(k), "NP") && !kids[..i].iter().any(|p| top_label(p) == "NP");
                if child_subject && skip_it {
                    pos += n;
                    i += 1;
                    continue;
                }
                // the object of a place PP, and the head NP of an NP that
                // continues with PPs of its own ("the city of Munich")
                let child_locative = (place_pp && i == 1 && matches!(top_label(k), "NP")) || (locative && is_np && matches!(top_label(k), "NP" | "NML"));
                let child_motion = (motion_pp && i == 1 && matches!(top_label(k), "NP")) || (motion && is_np && matches!(top_label(k), "NP" | "NML"));
                walk(k, pos, sentence, child_subject, child_locative, child_motion, child_subject && is_speech, out);
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
        span: None,
        conjunct: false,
        locative: false,
        possessor: false,
        speaker: false,
        motion: false,
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
        span: None,
        conjunct: false,
        locative: false,
        possessor: false,
        speaker: false,
        motion: false,
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
        _ if core.iter().any(|l| l.tag == "CC" && matches!(l.word.to_lowercase().as_str(), "and" | "both")) => Number::Plural,
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
        span: None,
        conjunct: false,
        locative: false,
        possessor: false,
        speaker: false,
        motion: false,
        candidates: Vec::new(),
    })
}

// Fill in antecedent candidates for every pronoun, looking back over the
// mentions of this and earlier sentences (in document order).
const FREQUENCY_WEIGHT: f32 = 1.0;
// a he/she resolution teaches a name its gender only when it is this
// clear of the next candidate
const LEARN_MARGIN: f32 = 0.5;

const FEMALE_NOUNS: &[&str] = &[
    "woman", "lady", "girl", "mother", "sister", "daughter", "wife", "widow", "aunt", "niece", "grandmother", "queen", "princess",
    "actress", "waitress", "nun", "bride", "stepmother", "stepdaughter", "madam", "mrs", "miss", "mistress", "maid", "heroine",
];
const MALE_NOUNS: &[&str] = &[
    "man", "gentleman", "boy", "father", "brother", "son", "husband", "widower", "uncle", "nephew", "grandfather", "king", "prince",
    "actor", "waiter", "monk", "groom", "stepfather", "stepson", "sir", "mr", "lad", "fellow", "chap", "guy", "hero",
];
const PERSON_NOUNS: &[&str] = &[
    "person", "child", "kid", "baby", "friend", "doctor", "visitor", "client", "stranger", "servant", "patient", "neighbour",
    "neighbor", "companion", "colleague", "partner", "relative", "cousin", "parent", "spouse", "twin", "officer", "detective",
    "inspector", "butler", "blacksmith", "coroner", "major", "general", "squire", "heir", "gambler", "pauper", "correspondent",
    "master", "villager", "people", "folks", "student", "teacher", "nurse", "judge", "author", "agent", "assistant", "chief",
    "boss", "chef", "pilot", "resident", "president",
];

/// Gender a common noun implies for a person ("widow", "stepfather").
pub(crate) fn gender_of_noun(head: &str) -> Option<Gender> {
    let h = head.to_lowercase();
    if FEMALE_NOUNS.contains(&h.as_str()) {
        Some(Gender::Female)
    } else if MALE_NOUNS.contains(&h.as_str()) {
        Some(Gender::Male)
    } else {
        None
    }
}

pub(crate) fn is_person_noun(head: &str) -> bool {
    let h = head.to_lowercase();
    gender_of_noun(&h).is_some() || PERSON_NOUNS.contains(&h.as_str()) || ["er", "or", "ist", "ian", "ess"].iter().any(|s| h.len() > s.len() + 2 && h.ends_with(s))
}
const RECENCY_WEIGHT: f32 = 0.5;
// "Smith's room": the possessor is said in passing
const POSSESSOR_PENALTY: f32 = 0.5;
// "Holmes answered": the speaker is who the next sentence is about
const SPEAKER_BONUS: f32 = 0.5;

pub fn resolve(all: &mut [Mention]) {
    let snapshot: Vec<Mention> = all.to_vec();
    // genders learned from the name list and from earlier resolutions,
    // keyed by lowercased mention text
    let mut known: Vec<(String, String, Gender)> = Vec::new();
    for c in &snapshot {
        if c.kind == Kind::Name {
            if let Some(g) = gender_of_name(&c.text) {
                known.push((c.text.to_lowercase(), c.text.clone(), g));
            }
        }
    }
    // names that are places everywhere in the text: once after a
    // preposition of place, or twice after to/from, and never with a
    // title or a gender ("Stoke Moran", "Leatherhead")
    let mut places: Vec<String> = Vec::new();
    for c in &snapshot {
        if c.kind != Kind::Name || !(c.locative || c.motion) || places.iter().any(|p| same_name(p, &c.text)) {
            continue;
        }
        let gendered = snapshot.iter().any(|o| o.kind == Kind::Name && same_name(&o.text, &c.text) && (title_of(&o.text).is_some() || gender_of_name(&o.text).is_some()));
        let motions = snapshot.iter().filter(|o| o.kind == Kind::Name && o.motion && same_name(&o.text, &c.text)).count();
        if !gendered && (c.locative || motions >= 2) {
            places.push(c.text.clone());
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
            if j >= i || c.kind == Kind::Pronoun || c.conjunct {
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
            // "Smith" takes the gender of "Alice Smith"
            let have = known
                .iter()
                .find(|(k, _, _)| *k == key)
                .or_else(|| known.iter().find(|(_, orig, _)| c.kind == Kind::Name && same_name(orig, &c.text)))
                .map(|(_, _, g)| *g);
            let mut score = 0.0f32;
            match (animate, c.kind) {
                (true, Kind::Name) => {
                    score += 2.0;
                    match (want, have) {
                        (Some(w), Some(h)) if w == h => score += 2.0,
                        (Some(_), Some(_)) => continue,
                        // "in Evanston": a place unless we know otherwise
                        _ if c.locative || places.iter().any(|p| same_name(p, &c.text)) => continue,
                        _ => {}
                    }
                }
                // "the widow", "my stepfather": a person whose gender the
                // noun gives away; "the doctor": a person; "the time": not
                (true, Kind::Nominal) => match (want, gender_of_noun(&c.head)) {
                    (Some(w), Some(h)) if w == h => score += 1.0,
                    (Some(_), Some(_)) => continue,
                    _ if is_person_noun(&c.head) => score += 0.5,
                    _ => {}
                },
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
            if c.possessor {
                score -= POSSESSOR_PENALTY;
            }
            if c.speaker {
                score += SPEAKER_BONUS;
            }
            // a name mentioned several times is what the text is about
            if c.kind == Kind::Name && animate {
                let times = snapshot[..i].iter().filter(|o| o.kind == Kind::Name && same_name(&o.text, &c.text)).count();
                score += FREQUENCY_WEIGHT * (times.min(5) as f32 - 1.0);
            }
            let distance = (m.sentence - c.sentence) as f32;
            score -= 0.5 * distance;
            if c.sentence == m.sentence && !reflexive {
                // closer to the pronoun is better, within the sentence
                score += 0.25 + RECENCY_WEIGHT * c.start as f32 / m.start.max(1) as f32;
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
                span: c.span,
            });
        }
        cands.sort_by(|a, b| b.score.total_cmp(&a.score));
        // one entry per name: "Smith" after "Alice Smith" is the same person
        let mut seen: Vec<String> = Vec::new();
        cands.retain(|c| {
            if seen.iter().any(|k| k.to_lowercase() == c.text.to_lowercase() || same_name(k, &c.text)) {
                false
            } else {
                seen.push(c.text.clone());
                true
            }
        });
        cands.truncate(5);
        // remember the gender a clear he/she resolution implies for a name
        let clear = cands.len() == 1 || cands.len() >= 2 && cands[0].score - cands[1].score >= LEARN_MARGIN;
        if let (Some(w), Some(best), true) = (want, cands.first(), clear) {
            let key = best.text.to_lowercase();
            if !known.iter().any(|(k, _, _)| *k == key)
                && snapshot.iter().any(|c| c.kind == Kind::Name && c.text.to_lowercase() == key)
                && !places.iter().any(|p| same_name(p, &best.text))
            {
                known.push((key, best.text.clone(), w));
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
/// Words joined with spaces, hyphen runs closed up ("owner-run").
pub fn join_words<S: AsRef<str>>(words: &[S]) -> String {
    let words: Vec<&str> = words.iter().map(|w| w.as_ref()).collect();
    words.join(" ").replace(" - ", "-").replace(" 's", "'s").replace(" n't", "n't")
}

// Byte spans for every mention and candidate, from the word spans of each
// sentence (as text::offsets gives them).
pub fn attach_spans(all: &mut [Mention], offsets: &[Vec<(usize, usize)>]) {
    let span = |sentence: usize, start: usize, end: usize| -> Option<(usize, usize)> {
        let words = offsets.get(sentence)?;
        Some((words.get(start)?.0, words.get(end.checked_sub(1)?)?.1))
    };
    for m in all.iter_mut() {
        m.span = span(m.sentence, m.start, m.end);
        for c in m.candidates.iter_mut() {
            c.span = span(c.sentence, c.start, c.end);
        }
    }
}

const NOT_NAME_WORDS: &[&str] = &["the", "a", "an", "this", "that", "these", "those", "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "lady", "lord"];

pub(crate) fn name_words(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || c == '-')
        .filter(|w| w.chars().next().is_some_and(|c| c.is_uppercase()))
        .map(|w| w.trim_matches('.').to_lowercase())
        .filter(|w| !w.is_empty() && !NOT_NAME_WORDS.contains(&w.as_str()))
        .collect()
}

fn title_of(text: &str) -> Option<String> {
    let first = text.split_whitespace().next()?.trim_matches('.').to_lowercase();
    matches!(first.as_str(), "mr" | "mrs" | "ms" | "miss" | "dr" | "prof" | "sir" | "lady" | "lord").then_some(first)
}

/// Do two mention texts look like the same name: they share a capitalized
/// word ("Alice Smith" and "Smith", "Mrs. Smith" and "Alice Smith"), and
/// nothing says they are two people: a gender known for both that
/// differs, or "Mrs." against "Miss".
pub fn same_name(a: &str, b: &str) -> bool {
    let wa = name_words(a);
    let wb = name_words(b);
    if !wa.iter().any(|w| wb.contains(w)) {
        return false;
    }
    if let (Some(x), Some(y)) = (gender_of_name(a), gender_of_name(b)) {
        if x != y {
            return false;
        }
    }
    // "Miss Roylott" and "Dr. Roylott" are two people; "Mr." and "Dr."
    // may be one
    match (title_of(a), title_of(b)) {
        (Some(x), Some(y)) if x != y => !(matches!(x.as_str(), "miss" | "mrs" | "ms") || matches!(y.as_str(), "miss" | "mrs" | "ms")),
        _ => true,
    }
}

/// How far the best candidate of a pronoun is ahead of the next one that
/// is not the same name. None without candidates, infinite with one.
pub fn margin(m: &Mention) -> Option<f32> {
    let best = m.candidates.first()?;
    let next = m.candidates[1..]
        .iter()
        .find(|c| !same_name(&c.text, &best.text) && c.text.to_lowercase() != best.text.to_lowercase());
    Some(next.map_or(f32::INFINITY, |c| best.score - c.score))
}

/// The bracket to put after a pronoun: its best antecedent, or, when the
/// margin is below `min_margin`, up to `alternatives` candidates joined
/// with "or" ("Smith or Jones"); None when nothing should be shown.
pub fn bracket(m: &Mention, min_margin: f32, alternatives: usize) -> Option<String> {
    let best = m.candidates.first()?;
    if best.text.to_lowercase() == m.text.to_lowercase() {
        return None;
    }
    if margin(m).is_some_and(|g| g >= min_margin) {
        return Some(best.text.clone());
    }
    if alternatives < 2 {
        return None;
    }
    // candidates close enough to the best one to be worth listing
    let mut names: Vec<&str> = Vec::new();
    for c in &m.candidates {
        if best.score - c.score > min_margin || names.len() == alternatives {
            break;
        }
        if !names.iter().any(|n| same_name(n, &c.text) || n.to_lowercase() == c.text.to_lowercase()) {
            names.push(&c.text);
        }
    }
    (names.len() >= 2).then(|| names.join(" or "))
}

// Words of every sentence with each resolved pronoun followed by its best
// antecedent in brackets: "she [Alice] would call him [Bob]". Pronouns whose
// margin is below `min_margin` are left as they are, or get a list of up
// to `alternatives` candidates: "he [Smith or Jones]".
pub fn annotate(sentences: &[Vec<String>], all: &[Mention], min_margin: f32, alternatives: usize) -> Vec<String> {
    let mut out = Vec::new();
    for (si, words) in sentences.iter().enumerate() {
        let mut pieces: Vec<String> = words.clone();
        for m in all.iter().filter(|m| m.sentence == si && m.kind == Kind::Pronoun) {
            if let Some(b) = bracket(m, min_margin, alternatives) {
                pieces[m.start] = format!("{} [{}]", words[m.start], b);
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
        let ann = annotate(&words, &all, 0.0, 1);
        assert_eq!(ann[1], "They [The engineers] liked it [the report] .");
        // each pronoun has one candidate, so any threshold keeps them
        let strict = annotate(&words, &all, 100.0, 1);
        assert_eq!(strict, ann);
        let ents = entities(&all);
        let report = ents.iter().find(|(k, _)| k == "the report").unwrap();
        assert_eq!(report.1, vec![0, 1]);
        let trees2 = [
            "(S (NP (DT The) (NN engineer)) (VP (VBD read) (NP (DT the) (NN report))) (. .))",
            "(S (NP (PRP It)) (VP (VBD was) (ADJP (JJ long))) (. .))",
        ];
        let all2 = doc(&trees2);
        let it = all2.iter().find(|m| m.text == "It").unwrap();
        assert_eq!(it.candidates.len(), 2);
        let g = margin(it).unwrap();
        assert!(g > 0.0 && g < 1.0, "{g}");
        let words2: Vec<Vec<String>> = trees2
            .iter()
            .map(|t| crate::treebank::leaves(&parse(t).unwrap()).into_iter().map(|(_, w)| w).collect())
            .collect();
        assert_eq!(annotate(&words2, &all2, 1.0, 1)[1], "It was long .");
        assert_eq!(annotate(&words2, &all2, 1.0, 2)[1], "It [The engineer or the report] was long .");
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
    fn margin_ignores_other_forms_of_the_same_name() {
        let all = doc(&[
            "(S (NP (NNP Alice) (NNP Smith)) (VP (VBD met) (NP (NN friend))) (. .))",
            "(S (NP (NNP Smith)) (VP (VBD left)) (. .))",
            "(S (NP (PRP She)) (VP (VBD smiled)) (. .))",
        ]);
        let she = all.iter().find(|m| m.text == "She").unwrap();
        let names: Vec<&str> = she.candidates.iter().map(|c| c.text.as_str()).collect();
        // the two forms of Smith are merged into one candidate
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(same_name(names[0], "Smith"));
        let g = margin(she).unwrap();
        let best = she.candidates[0].score;
        let friend = she.candidates.iter().find(|c| c.text == "friend").unwrap().score;
        assert_eq!(g, best - friend, "{names:?}");
        assert!(same_name("Mrs. Smith", "Alice Smith"));
        assert!(!same_name("The report", "the Report card"));
    }

    #[test]
    fn spans_come_from_word_offsets() {
        let mut all = doc(&[
            "(S (NP (NNP Alice)) (VP (VBD left)) (. .))",
            "(S (NP (PRP She)) (VP (VBD smiled)) (. .))",
        ]);
        let offs = vec![vec![(0, 5), (6, 10), (10, 11)], vec![(12, 15), (16, 22), (22, 23)]];
        attach_spans(&mut all, &offs);
        let she = all.iter().find(|m| m.text == "She").unwrap();
        assert_eq!(she.span, Some((12, 15)));
        assert_eq!(she.candidates[0].span, Some((0, 5)));
    }

    #[test]
    fn flat_coordination_is_plural_with_its_names() {
        let m = mentions(
            &parse("(S (NP (NNP John) (CC and) (NNP Mary)) (VP (VBD bought) (NP (DT a) (NN house))) (. .))").unwrap(),
            0,
        );
        let texts: Vec<(&str, Kind, Number)> = m.iter().map(|x| (x.text.as_str(), x.kind, x.number)).collect();
        assert_eq!(
            texts,
            vec![
                ("John and Mary", Kind::Nominal, Number::Plural),
                ("John", Kind::Name, Number::Singular),
                ("Mary", Kind::Name, Number::Singular),
                ("a house", Kind::Nominal, Number::Singular),
            ]
        );
        assert!(m[1].conjunct && m[2].conjunct && !m[0].conjunct);
    }

    #[test]
    fn place_names_are_not_people() {
        let all = doc(&[
            "(S (NP (PRP He)) (VP (VBD grew) (PRT (RP up)) (PP (IN in) (NP (NNP Evanston)))) (. .))",
            "(S (NP (PRP$ His) (NN pottery)) (VP (VBZ is) (ADJP (JJ famous))) (. .))",
        ]);
        let evanston = all.iter().find(|m| m.text == "Evanston").unwrap();
        assert!(evanston.locative);
        let his = all.iter().find(|m| m.text == "His").unwrap();
        assert!(his.candidates.is_empty(), "{:?}", his.candidates);
    }

    #[test]
    fn frequent_names_and_nearby_names_rank_higher() {
        // Smith is named twice (once as "Alice Smith"), Bob once
        let all = doc(&[
            "(S (NP (NNP Alice) (NNP Smith)) (VP (VBD met) (NP (NNP Carol))) (. .))",
            "(S (NP (NNP Smith)) (VP (VBD left)) (. .))",
            "(S (NP (NNP Carol)) (VP (VBD said) (SBAR (S (NP (PRP she)) (VP (VBD smiled))))) (. .))",
        ]);
        let she = all.iter().find(|m| m.text == "she").unwrap();
        let names: Vec<&str> = she.candidates.iter().map(|c| c.text.as_str()).collect();
        // one entry per person, Carol first: same sentence and closest
        assert_eq!(names.len(), 2, "{names:?}");
        assert_eq!(names[0], "Carol");
        assert!(same_name(names[1], "Smith"));
        let all = doc(&[
            "(S (NP (NNP Alice)) (VP (VBD met) (NP (NNP Carol)) (PP (IN on) (NP (NNP Monday)))) (. .))",
            "(S (NP (NNP Alice)) (VP (VBD called) (NP (NNP Dana))) (. .))",
            "(S (NP (PRP She)) (VP (VBD smiled)) (. .))",
        ]);
        let she = all.iter().find(|m| m.text == "She").unwrap();
        assert_eq!(she.candidates[0].text, "Alice");
    }

    #[test]
    fn short_name_forms_share_the_gender() {
        let all = doc(&[
            "(S (NP (NNP Alice) (NNP Smith)) (VP (VBD met) (NP (NNP Angeloz))) (. .))",
            "(S (NP (NNP Smith)) (VP (VBD left)) (. .))",
            "(S (NP (PRP She)) (VP (VBD thanked) (NP (PRP him))) (. .))",
        ]);
        let she = all.iter().find(|m| m.text == "She").unwrap();
        assert!(same_name(&she.candidates[0].text, "Smith"));
        let him = all.iter().find(|m| m.text == "him").unwrap();
        assert_eq!(him.candidates[0].text, "Angeloz");
    }

    #[test]
    fn another_title_is_another_person() {
        assert!(same_name("Dr. Grimesby Roylott", "Dr. Roylott"));
        assert!(!same_name("Miss Roylott", "Dr. Roylott"));
        assert!(!same_name("Mrs. Stoner", "Miss Stoner"));
        let all = doc(&[
            "(S (NP (NNP Dr.) (NNP Roylott)) (VP (VBD met) (NP (NNP Miss) (NNP Roylott))) (. .))",
            "(S (NP (PRP She)) (VP (VBD left)) (. .))",
        ]);
        let she = all.iter().find(|m| m.text == "She").unwrap();
        assert_eq!(she.candidates[0].text, "Miss Roylott");
    }

    #[test]
    fn expletive_it_is_not_a_mention() {
        let all = doc(&["(S (NP (PRP It)) (VP (VBD was) (NP (PRP$ his) (NN custom)) (S (VP (TO to) (VP (VB smoke))))) (. .))"]);
        assert!(all.iter().all(|m| m.text != "It"));
        let all = doc(&["(S (NP (PRP It)) (VP (VBD was) (ADJP (JJ late))) (. .))"]);
        assert!(all.iter().any(|m| m.text == "It"));
    }

    #[test]
    fn a_place_once_is_a_place_everywhere() {
        let all = doc(&[
            "(S (NP (PRP We)) (VP (VBD drove) (PP (TO to) (NP (NNP Leatherhead)))) (. .))",
            "(S (NP (PRP We)) (VP (VBD came) (PP (IN from) (NP (NNP Leatherhead)))) (. .))",
            "(S (NP (PRP$ My) (NN companion)) (VP (VBD sat) (ADVP (RB silent))) (. .))",
            "(S (NP (NNP Leatherhead)) (VP (VBD was) (ADJP (JJ quiet))) (. .))",
            "(S (NP (PRP He)) (VP (VBD started)) (. .))",
        ]);
        let he = all.iter().find(|m| m.text == "He").unwrap();
        assert_eq!(he.candidates[0].text, "My companion");
    }

    #[test]
    fn possessor_and_speaker_are_marked() {
        let all = doc(&[
            "(S (NP (NNP Holmes)) (VP (VBD answered)) (. .))",
            "(S (NP (PRP We)) (VP (MD will) (VP (VB enter) (NP (NP (NNP Roylott) (POS 's)) (NN room)))) (. .))",
            "(S (NP (PRP He)) (VP (VBD lit) (NP (DT the) (NN lamp))) (. .))",
        ]);
        let holmes = all.iter().find(|m| m.text == "Holmes").unwrap();
        assert!(holmes.speaker && holmes.subject);
        let roylott = all.iter().find(|m| m.text == "Roylott").unwrap();
        assert!(roylott.possessor && !roylott.subject);
        let room = all.iter().find(|m| m.text == "Roylott's room").unwrap();
        assert_eq!(room.head, "room");
        let he = all.iter().find(|m| m.text == "He").unwrap();
        assert_eq!(he.candidates[0].text, "Holmes");
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
