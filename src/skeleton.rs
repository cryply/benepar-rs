// The skeleton of a sentence: its events (verbs) with their arguments,
// read off the tree with head rules, kept apart from the words that fill
// the slots. "Alice sent Bob the report" gives
//
//   sent  subject=Alice  indirect=Bob  object=the report
//
// Roles are grammatical, normalized for voice: in "the report was sent by
// Alice" the report is still the object and Alice the subject. Nothing
// here knows what "send" means; that is a lexicon's job.
//
// What is handled: auxiliary chains and negation, passives, verb
// coordination (one subject, two events), prepositional arguments, clause
// arguments (the inner event is an argument of the outer one), control
// ("told Alice to leave": Alice leaves; "wanted to leave": the subject
// leaves), relative clauses (the noun is the subject of a subject
// relative), copulas with their complement. Not handled: inversion,
// object relatives, ellipsis.

use crate::mentions::Mention;
use crate::tree::Node;
use serde::Serialize;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Arg {
    pub role: String,
    // words of the whole argument phrase
    pub start: usize,
    pub end: usize,
    // the head word
    pub head: usize,
    pub text: String,
    // index into the mentions of the sentence, when the argument is one
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mention: Option<usize>,
    // for clause arguments: the inner event
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Event {
    pub sentence: usize,
    // index of the main verb word
    pub verb: usize,
    pub word: String,
    pub tag: String,
    // auxiliaries, modals and particles, in order
    pub aux: Vec<String>,
    pub negated: bool,
    pub passive: bool,
    pub args: Vec<Arg>,
}

// Collapsed unary chains ("S::VP") expanded into nested nodes.
pub(crate) fn expand(t: &Node) -> Node {
    match t {
        Node::Leaf(..) => t.clone(),
        Node::Phrase(_, kids) => {
            let kids: Vec<Node> = kids.iter().map(expand).collect();
            let labels = t.labels();
            let mut node = Node::Phrase(labels.last().copied().unwrap_or("").to_string(), kids);
            for l in labels.iter().rev().skip(1) {
                node = Node::Phrase(l.to_string(), vec![node]);
            }
            node
        }
    }
}

fn label(t: &Node) -> &str {
    match t {
        Node::Phrase(l, _) => l.as_str(),
        Node::Leaf(..) => "",
    }
}

fn kids(t: &Node) -> &[Node] {
    match t {
        Node::Phrase(_, k) => k,
        Node::Leaf(..) => &[],
    }
}

fn size(t: &Node) -> usize {
    match t {
        Node::Leaf(..) => 1,
        Node::Phrase(_, k) => k.iter().map(size).sum(),
    }
}

fn words(t: &Node, out: &mut Vec<(String, String)>) {
    match t {
        Node::Leaf(tag, w) => out.push((tag.clone(), w.clone())),
        Node::Phrase(_, k) => k.iter().for_each(|c| words(c, out)),
    }
}

fn is_clause(l: &str) -> bool {
    matches!(l, "S" | "SINV" | "SQ")
}

fn is_verb_tag(t: &str) -> bool {
    t.starts_with("VB") || t == "MD"
}

fn is_copula(w: &str) -> bool {
    matches!(w.to_lowercase().as_str(), "be" | "am" | "is" | "are" | "was" | "were" | "been" | "being" | "'s" | "'re" | "'m")
}

fn stem(w: &str) -> String {
    let w = w.to_lowercase();
    for suffix in ["ing", "ed", "es", "s"] {
        if let Some(s) = w.strip_suffix(suffix).filter(|s| s.len() >= 3) {
            return s.to_string();
        }
    }
    w
}

// Verbs whose noun phrase is a predicate about the subject, not an
// object: "became president", "remained a mystery", "seems a fool".
fn links_np(w: &str) -> bool {
    matches!(stem(w).as_str(), "becom" | "become" | "became" | "remain" | "seem" | "appear" | "stay")
}

// Verbs whose adjective phrase is a predicate about the subject: the
// above plus "felt odd", "looked tired", "got angry", "turned red".
fn links_adjp(w: &str) -> bool {
    links_np(w) || matches!(stem(w).as_str(), "feel" | "felt" | "look" | "sound" | "turn" | "prov" | "prove" | "grow" | "grew" | "grown" | "get" | "got" | "gotten" | "smell" | "taste")
}

const TIME_WORDS: &[&str] = &[
    "yesterday", "today", "tomorrow", "tonight", "now", "then", "year", "years", "month", "months", "week", "weeks", "day", "days",
    "hour", "hours", "minute", "minutes", "night", "nights", "morning", "evening", "afternoon", "quarter", "decade", "decades", "century",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december", "ago", "time", "times", "season", "spring", "summer", "fall", "winter",
    "moment", "period", "weekend",
];

const TIME_ADVERBS: &[&str] = &[
    "yesterday", "today", "tomorrow", "tonight", "now", "then", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
];

// A bare noun phrase that tells when: "yesterday", "last year", "two
// weeks ago", "every Monday". "the day" and "a year" are things ("spent
// the day"), unless they end in "ago".
fn is_temporal(np: &Node, head: usize, start: usize) -> bool {
    let mut w = Vec::new();
    words(np, &mut w);
    let (tag, word) = &w[head - start];
    let word = word.to_lowercase();
    let first = w.first().map(|(_, x)| x.to_lowercase()).unwrap_or_default();
    let last = w.last().map(|(_, x)| x.to_lowercase()).unwrap_or_default();
    if last == "ago" || TIME_ADVERBS.contains(&word.as_str()) {
        return true;
    }
    let article = matches!(first.as_str(), "the" | "a" | "an");
    (tag.starts_with("NN") || tag == "CD") && !article && TIME_WORDS.contains(&word.as_str())
}

/// Index of the head word of a noun phrase: the last noun before any PP,
/// clause or punctuation; otherwise the last word of the first base NP.
pub fn np_head(t: &Node, start: usize) -> usize {
    let k = kids(t);
    // an NP made of NPs: the first one is the head ("the report of Alice")
    let first_np = k.iter().enumerate().find(|(_, c)| matches!(label(c), "NP" | "NML" | "NX" | "WHNP"));
    let has_noun = k.iter().any(|c| matches!(c, Node::Leaf(t, _) if t.starts_with("NN") || t == "PRP"));
    if let (Some((i, first_np)), false) = (first_np, has_noun) {
        let off: usize = k[..i].iter().map(size).sum();
        return np_head(first_np, start + off);
    }
    let mut pos = start;
    let mut head = None;
    for c in k {
        match c {
            Node::Leaf(tag, _) => {
                if tag.starts_with("NN") || matches!(tag.as_str(), "PRP" | "CD" | "EX" | "WP" | "DT" | "JJ" | "JJR" | "JJS" | "VBG" | "VBN" | "FW") {
                    let noun = tag.starts_with("NN") || matches!(tag.as_str(), "PRP" | "CD" | "EX" | "WP");
                    if noun || head.is_none() {
                        head = Some(pos);
                    }
                }
            }
            Node::Phrase(l, _) => {
                if matches!(l.as_str(), "PP" | "SBAR" | "S" | "VP" | "RRC" | "PRN" | "ADJP" | "ADVP") && head.is_some() {
                    break;
                }
                if matches!(l.as_str(), "NML" | "NX" | "QP") {
                    head = Some(np_head(c, pos));
                }
            }
        }
        pos += size(c);
    }
    head.unwrap_or(start + size(t).saturating_sub(1))
}

struct Group<'a> {
    aux: Vec<String>,
    verb: Option<(usize, String, String)>,
    negated: bool,
    // the innermost VP, whose other children are the arguments
    vp: &'a Node,
    vp_start: usize,
}

// The verb groups of a VP: one per coordinated conjunct, each with its
// auxiliary chain followed down to the main verb.
fn groups<'a>(vp: &'a Node, start: usize, out: &mut Vec<Group<'a>>) {
    let k = kids(vp);
    let vps: Vec<usize> = k.iter().enumerate().filter(|(_, c)| label(c) == "VP").map(|(i, _)| i).collect();
    let coordinated = vps.len() > 1 && k.iter().any(|c| matches!(c, Node::Leaf(t, _) if t == "CC"));
    if coordinated {
        for &i in &vps {
            let off: usize = k[..i].iter().map(size).sum();
            groups(&k[i], start + off, out);
        }
        return;
    }
    let mut g = Group { aux: Vec::new(), verb: None, negated: false, vp, vp_start: start };
    let mut pos = start;
    for (i, c) in k.iter().enumerate() {
        match c {
            Node::Leaf(tag, w) => {
                if is_verb_tag(tag) || tag == "TO" {
                    if vps.is_empty() && is_verb_tag(tag) {
                        if g.verb.is_none() {
                            g.verb = Some((pos, tag.clone(), w.clone()));
                        }
                    } else {
                        g.aux.push(w.clone());
                    }
                } else if tag == "RB" && matches!(w.to_lowercase().as_str(), "not" | "n't" | "never") {
                    g.negated = true;
                }
            }
            Node::Phrase(l, _) if l == "VP" && Some(i) == vps.first().copied() => {
                // the chain continues inside
                let mut inner = Vec::new();
                groups(c, pos, &mut inner);
                for mut h in inner {
                    let mut aux = g.aux.clone();
                    aux.append(&mut h.aux);
                    h.aux = aux;
                    h.negated |= g.negated;
                    out.push(h);
                }
                return;
            }
            Node::Phrase(l, _) if l == "PRT" => {
                let mut w = Vec::new();
                words(c, &mut w);
                g.aux.extend(w.into_iter().map(|(_, x)| x));
            }
            _ => {}
        }
        pos += size(c);
    }
    if g.verb.is_some() {
        out.push(g);
    }
}

struct Builder<'a> {
    sentence: usize,
    mentions: &'a [Mention],
    events: Vec<Event>,
}

impl Builder<'_> {
    fn arg(&self, role: &str, node: &Node, start: usize) -> Arg {
        let end = start + size(node);
        let head = if matches!(label(node), "NP" | "NML" | "WHNP") { np_head(node, start) } else { end - 1 };
        let mut w = Vec::new();
        words(node, &mut w);
        let text: Vec<&str> = w.iter().map(|(_, x)| x.as_str()).collect();
        let mention = self
            .mentions
            .iter()
            .enumerate()
            .filter(|(_, m)| m.sentence == self.sentence && m.start == start && m.end <= end)
            .max_by_key(|(_, m)| m.end)
            .map(|(i, _)| i);
        let text = match mention {
            Some(i) => self.mentions[i].text.clone(),
            None => crate::mentions::join_words(&text),
        };
        Arg { role: role.to_string(), start, end, head, text, mention, event: None }
    }

    // Events of a clause node. `subject` is the subject to use when the
    // clause has none of its own (control, relatives).
    fn clause(&mut self, node: &Node, start: usize, subject: Option<Arg>) -> Vec<usize> {
        let k = kids(node);
        let mut pos = start;
        let mut subj = None;
        let mut made = Vec::new();
        for c in k {
            match label(c) {
                "NP" if subj.is_none() && !made.iter().any(|_| true) => {
                    subj = Some(self.arg("subject", c, pos));
                }
                "VP" => {
                    let s = subj.clone().or_else(|| subject.clone());
                    made.extend(self.vp(c, pos, s));
                }
                "S" | "SINV" | "SQ" | "SBAR" => {
                    // a clause of its own inside this one ("S CC S")
                    made.extend(self.any(c, pos, subject.clone()));
                }
                _ => {
                    made.extend(self.any(c, pos, None));
                }
            }
            pos += size(c);
        }
        if let Some(s) = subj {
            // the subject itself may hold a relative clause
            let off: usize = k.iter().take_while(|c| label(c) != "NP").map(size).sum();
            let np = k.iter().find(|c| label(c) == "NP").unwrap();
            made.extend(self.np(np, start + off, &s));
        }
        made
    }

    // Relative clauses and other clauses inside a noun phrase.
    fn np(&mut self, node: &Node, start: usize, this: &Arg) -> Vec<usize> {
        let mut made = Vec::new();
        let mut pos = start;
        for c in kids(node) {
            match label(c) {
                "SBAR" => {
                    let has_wh = kids(c).iter().any(|x| label(x).starts_with("WH"));
                    let hint = if has_wh { Some(this.clone()) } else { None };
                    made.extend(self.any(c, pos, hint));
                }
                "VP" => {
                    // reduced relative: "the report sent by Alice"
                    made.extend(self.vp(c, pos, Some(this.clone())));
                }
                "NP" | "PP" | "S" => made.extend(self.any(c, pos, None)),
                _ => {}
            }
            pos += size(c);
        }
        made
    }

    fn vp(&mut self, vp: &Node, start: usize, subject: Option<Arg>) -> Vec<usize> {
        let mut gs = Vec::new();
        groups(vp, start, &mut gs);
        let mut made = Vec::new();
        for g in gs {
            let Some((vi, tag, word)) = g.verb.clone() else { continue };
            let passive = tag == "VBN" && g.aux.iter().any(|a| is_copula(a) || matches!(a.to_lowercase().as_str(), "get" | "gets" | "got" | "getting" | "gotten"));
            let copula = is_copula(&word);
            let mut args = Vec::new();
            let mut pending: Vec<(usize, &Node, usize, Option<Arg>)> = Vec::new();
            let mut nps = 0;
            let mut pos = g.vp_start;
            for c in kids(g.vp) {
                match label(c) {
                    "NP" => {
                        let mut a = self.arg("object", c, pos);
                        if is_temporal(c, a.head, pos) {
                            a.role = "time".to_string();
                        } else if copula || links_np(&word) {
                            a.role = "complement".to_string();
                        } else {
                            nps += 1;
                        }
                        args.push(a);
                    }
                    "PP" => {
                        if let Some((prep, np, off)) = pp_parts(c) {
                            let role = format!("prep:{}", prep.to_lowercase());
                            args.push(self.arg(&role, np, pos + off));
                        }
                    }
                    "ADJP" if copula || links_adjp(&word) => args.push(self.arg("complement", c, pos)),
                    "S" | "SBAR" | "SQ" | "SINV" => {
                        // the inner clause's subject when it has none: the
                        // object if there is one, else our subject
                        let ctl = args.iter().rev().find(|a| a.role == "object").cloned().or_else(|| subject.clone());
                        pending.push((args.len(), c, pos, ctl));
                        // the complementizer names the role: "clause:because"
                        let role = match kids(c).first() {
                            Some(Node::Leaf(t, w)) if matches!(t.as_str(), "IN" | "WRB") && w.to_lowercase() != "that" => format!("clause:{}", w.to_lowercase()),
                            _ => "clause".to_string(),
                        };
                        args.push(Arg { role, start: pos, end: pos + size(c), head: pos, text: String::new(), mention: None, event: None });
                    }
                    _ => {}
                }
                pos += size(c);
            }
            // two bare objects: the first is the indirect one
            if let (2, Some(a)) = (nps, args.iter_mut().find(|a| a.role == "object")) {
                a.role = "indirect".to_string();
            }
            if let Some(mut s) = subject.clone() {
                s.role = "subject".to_string();
                args.insert(0, s);
            }
            if passive {
                for a in args.iter_mut() {
                    if a.role == "subject" {
                        a.role = "object".to_string();
                    } else if a.role == "prep:by" {
                        a.role = "subject".to_string();
                    }
                }
            }
            let id = self.events.len();
            self.events.push(Event { sentence: self.sentence, verb: vi, word, tag, aux: g.aux, negated: g.negated, passive, args });
            made.push(id);
            for (ai, node, pos, ctl) in pending {
                // the argument index shifted by the inserted subject
                let ai = ai + usize::from(subject.is_some());
                let inner = self.any(node, pos, ctl);
                if let Some(&first) = inner.first() {
                    let verb = self.events[first].verb;
                    let a = &mut self.events[id].args[ai];
                    a.event = Some(first);
                    a.head = verb;
                    a.text = format!("e{}", first + 1);
                }
            }
            // arguments with relative clauses of their own
            let arg_nodes: Vec<(&Node, usize)> = {
                let mut v = Vec::new();
                let mut pos = g.vp_start;
                for c in kids(g.vp) {
                    if label(c) == "NP" {
                        v.push((c, pos));
                    } else if let (true, Some((_, np, off))) = (label(c) == "PP", pp_parts(c)) {
                        v.push((np, pos + off));
                    }
                    pos += size(c);
                }
                v
            };
            for (n, p) in arg_nodes {
                let this = self.arg("subject", n, p);
                made.extend(self.np(n, p, &this));
            }
        }
        made
    }

    // Any node: clauses are built, everything else is searched for them.
    fn any(&mut self, node: &Node, start: usize, subject: Option<Arg>) -> Vec<usize> {
        let l = label(node);
        if is_clause(l) {
            return self.clause(node, start, subject);
        }
        if l == "SBAR" {
            let mut made = Vec::new();
            let mut pos = start;
            for c in kids(node) {
                if is_clause(label(c)) || label(c) == "SBAR" {
                    made.extend(self.any(c, pos, subject.clone()));
                }
                pos += size(c);
            }
            return made;
        }
        if l == "VP" {
            return self.vp(node, start, subject);
        }
        let mut made = Vec::new();
        let mut pos = start;
        for c in kids(node) {
            if let Node::Phrase(..) = c {
                made.extend(self.any(c, pos, None));
            }
            pos += size(c);
        }
        made
    }
}

// Preposition and noun phrase of a PP, with the NP's offset inside it.
fn pp_parts(pp: &Node) -> Option<(String, &Node, usize)> {
    let k = kids(pp);
    let prep = k.iter().find_map(|c| match c {
        Node::Leaf(t, w) if matches!(t.as_str(), "IN" | "TO" | "RP") => Some(w.clone()),
        _ => None,
    })?;
    let mut off = 0;
    for c in k {
        if matches!(label(c), "NP" | "NML" | "S" | "SBAR") {
            return Some((prep, c, off));
        }
        if label(c) == "PP" {
            let (p2, np, o2) = pp_parts(c)?;
            return Some((format!("{prep} {p2}"), np, off + o2));
        }
        off += size(c);
    }
    None
}

/// Events of one parsed sentence. `mentions` are its mentions, so that
/// arguments can point at them.
pub fn events(tree: &Node, sentence: usize, mentions: &[Mention]) -> Vec<Event> {
    let t = expand(tree);
    let mut b = Builder { sentence, mentions, events: Vec::new() };
    b.any(&t, 0, None);
    b.events
}

/// One line per event: "e1 sent subject=Alice indirect=Bob object=the report".
pub fn lines(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let mut s = format!("e{} {}", i + 1, e.word);
            if !e.aux.is_empty() {
                s.push_str(&format!(" [{}]", e.aux.join(" ")));
            }
            if e.negated {
                s.push_str(" not");
            }
            if e.passive {
                s.push_str(" passive");
            }
            for a in &e.args {
                s.push_str(&format!("  {}={}", a.role, a.text));
            }
            s
        })
        .collect()
}

// Surface relations for scoring: (verb index, head index) pairs for
// subjects and for direct objects, before voice normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Rel {
    Subject,
    Object,
}

pub fn relations(events: &[Event]) -> Vec<(Rel, usize, usize)> {
    let mut out = Vec::new();
    for e in events {
        for a in &e.args {
            let surface = match (e.passive, a.role.as_str()) {
                (false, "subject") | (true, "object") => Some(Rel::Subject),
                (false, "object") | (false, "indirect") => Some(Rel::Object),
                _ => None,
            };
            if let (Some(r), None) = (surface, a.event) {
                out.push((r, e.verb, a.head));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

// The same relations from a treebank tree with function tags and traces,
// as the gold standard. Word positions count the words that remain after
// empty elements are dropped. A traced subject is followed to its
// antecedent through the coindex; an unindexed empty subject is skipped.
pub fn gold_relations(raw: &Node) -> Vec<(Rel, usize, usize)> {
    // every constituent with a numeric index, with its head word
    let mut indexed: Vec<(String, usize)> = Vec::new();
    let mut out = Vec::new();
    collect_gold(raw, &mut 0, None, &mut indexed, &mut out);
    let mut rels = Vec::new();
    for (rel, verb, target) in out {
        let head = match target {
            GoldTarget::Head(h) => Some(h),
            GoldTarget::Trace(ix) => indexed.iter().find(|(i, _)| *i == ix).map(|(_, h)| *h),
        };
        if let Some(h) = head {
            rels.push((rel, verb, h));
        }
    }
    rels.sort();
    rels.dedup();
    rels
}

enum GoldTarget {
    Head(usize),
    Trace(String),
}

fn tags_of(label: &str) -> Vec<String> {
    label.split('-').skip(1).map(|t| t.split('=').next().unwrap_or("").to_string()).collect()
}

fn index_of(label: &str) -> Option<String> {
    let last = label.rsplit(['-', '=']).next()?;
    if !last.is_empty() && last.chars().all(|c| c.is_ascii_digit()) && label.contains(['-', '=']) {
        Some(last.to_string())
    } else {
        None
    }
}

// `governor` is the head of the noun phrase a relative clause modifies,
// so that a relative pronoun stands for that noun.
fn collect_gold(t: &Node, pos: &mut usize, governor: Option<usize>, indexed: &mut Vec<(String, usize)>, out: &mut Vec<(Rel, usize, GoldTarget)>) {
    let Node::Phrase(raw_label, k) = t else {
        if !matches!(t, Node::Leaf(tag, _) if tag == "-NONE-") {
            *pos += 1;
        }
        return;
    };
    let start = *pos;
    let base = crate::treebank::strip_label(raw_label);
    let clean = crate::treebank::clean(t);
    let own_head = match (&clean, base.as_str()) {
        (Some(c), "NP") => Some(np_head(c, start)),
        _ => None,
    };
    // positions of the children, computed first
    let mut child_pos = Vec::new();
    for c in k {
        child_pos.push(*pos);
        let g = match base.as_str() {
            "NP" => own_head,
            "SBAR" => governor,
            _ => None,
        };
        collect_gold(c, pos, g, indexed, out);
    }
    if let (Some(ix), Some(clean)) = (index_of(raw_label), &clean) {
        match (base.as_str(), governor) {
            ("WHNP", Some(h)) => indexed.push((ix, h)),
            ("NP" | "WHNP", _) => indexed.push((ix, np_head(clean, start))),
            _ => {}
        }
    }
    if !is_clause(&base) {
        return;
    }
    // subject and main verb of this clause
    let mut subject = None;
    let mut verb = None;
    let mut objects = Vec::new();
    for (c, &cp) in k.iter().zip(&child_pos) {
        let Node::Phrase(cl, _) = c else { continue };
        let cb = crate::treebank::strip_label(cl);
        if cb == "NP" && tags_of(cl).iter().any(|t| t == "SBJ") && subject.is_none() {
            subject = Some(gold_target(c, cp));
        }
        if cb == "VP" && verb.is_none() {
            let (v, objs) = gold_vp(c, cp);
            verb = v;
            objects = objs;
        }
    }
    if let (Some(v), Some(s)) = (verb, subject) {
        out.push((Rel::Subject, v, s));
        for o in objects {
            out.push((Rel::Object, v, o));
        }
    }
}

fn gold_target(np: &Node, start: usize) -> GoldTarget {
    let mut w = Vec::new();
    words(np, &mut w);
    if w.len() == 1 && w[0].0 == "-NONE-" {
        let trace = &w[0].1;
        match trace.rsplit('-').next() {
            Some(ix) if trace.contains('-') && ix.chars().all(|c| c.is_ascii_digit()) => GoldTarget::Trace(ix.to_string()),
            _ => GoldTarget::Trace(String::new()),
        }
    } else {
        match crate::treebank::clean(np) {
            Some(c) => GoldTarget::Head(np_head(&c, start)),
            None => GoldTarget::Trace(String::new()),
        }
    }
}

// Main verb of a gold VP (following the auxiliary chain) and the heads of
// its bare NP objects.
fn gold_vp(vp: &Node, start: usize) -> (Option<usize>, Vec<GoldTarget>) {
    let k = kids(vp);
    let mut pos = start;
    let mut verb = None;
    let mut objects = Vec::new();
    let inner = k.iter().enumerate().find(|(_, c)| crate::treebank::strip_label(label(c)) == "VP");
    for (i, c) in k.iter().enumerate() {
        match c {
            Node::Leaf(tag, _) => {
                if is_verb_tag(tag) && inner.is_none() && verb.is_none() {
                    verb = Some(pos);
                }
                if tag != "-NONE-" {
                    pos += 1;
                }
                continue;
            }
            Node::Phrase(l, _) => {
                let base = crate::treebank::strip_label(l);
                if base == "VP" && Some(i) == inner.map(|(j, _)| j) {
                    let (v, mut objs) = gold_vp(c, pos);
                    verb = v;
                    objects.append(&mut objs);
                } else if base == "NP" && tags_of(l).is_empty() && inner.is_none() {
                    objects.push(gold_target(c, pos));
                }
            }
        }
        pos += gold_size(c);
    }
    (verb, objects)
}

fn gold_size(t: &Node) -> usize {
    match t {
        Node::Leaf(tag, _) => usize::from(tag != "-NONE-"),
        Node::Phrase(_, k) => k.iter().map(gold_size).sum(),
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub subj_match: usize,
    pub subj_gold: usize,
    pub subj_pred: usize,
    pub obj_match: usize,
    pub obj_gold: usize,
    pub obj_pred: usize,
}

impl Counts {
    pub fn add(&mut self, gold: &[(Rel, usize, usize)], pred: &[(Rel, usize, usize)]) {
        for r in gold {
            let (m, g) = if r.0 == Rel::Subject { (&mut self.subj_match, &mut self.subj_gold) } else { (&mut self.obj_match, &mut self.obj_gold) };
            *g += 1;
            if pred.contains(r) {
                *m += 1;
            }
        }
        for r in pred {
            if r.0 == Rel::Subject {
                self.subj_pred += 1;
            } else {
                self.obj_pred += 1;
            }
        }
    }

    pub fn f1(m: usize, g: usize, p: usize) -> (f64, f64, f64) {
        let pr = 100.0 * m as f64 / p.max(1) as f64;
        let rc = 100.0 * m as f64 / g.max(1) as f64;
        let f = if pr + rc == 0.0 { 0.0 } else { 2.0 * pr * rc / (pr + rc) };
        (pr, rc, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mentions::mentions;
    use crate::treebank::parse;

    fn run(tree: &str) -> Vec<String> {
        let t = parse(tree).unwrap();
        let m = mentions(&t, 0);
        lines(&events(&t, 0, &m))
    }

    #[test]
    fn ditransitive_and_pronouns() {
        let l = run("(S (NP (NNP Alice)) (VP (VBD sent) (NP (NNP Bob)) (NP (DT the) (NN report))) (. .))");
        assert_eq!(l, vec!["e1 sent  subject=Alice  indirect=Bob  object=the report"]);
        let l = run("(S (NP (PRP He)) (VP (VBD read) (NP (PRP it)) (PP (IN at) (NP (NN home)))) (. .))");
        assert_eq!(l, vec!["e1 read  subject=He  object=it  prep:at=home"]);
    }

    #[test]
    fn passive_and_negation() {
        let l = run("(S (NP (DT The) (NN report)) (VP (VBD was) (RB not) (VP (VBN sent) (PP (IN by) (NP (NNP Alice))))) (. .))");
        assert_eq!(l, vec!["e1 sent [was] not passive  object=The report  subject=Alice"]);
        let l = run("(S (NP (NNP Bob)) (VP (MD will) (VP (VB have) (VP (VBN left)))) (. .))");
        assert_eq!(l, vec!["e1 left [will have]  subject=Bob"]);
    }

    #[test]
    fn coordination_control_and_relatives() {
        let l = run("(S (NP (NNP Bob)) (VP (VP (VBD read) (NP (PRP it))) (CC and) (VP (VBD told) (NP (NNP Alice)) (S (VP (TO to) (VP (VB leave)))))) (. .))");
        assert_eq!(
            l,
            vec![
                "e1 read  subject=Bob  object=it",
                "e2 told  subject=Bob  object=Alice  clause=e3",
                "e3 leave [to]  subject=Alice",
            ]
        );
        let l = run("(S (NP (NP (DT The) (NN man)) (SBAR (WHNP (WP who)) (S (VP (VBD called))))) (VP (VBD smiled)) (. .))");
        assert_eq!(l, vec!["e1 smiled  subject=The man", "e2 called  subject=The man"]);
        let l = run("(S (NP (PRP She)) (VP (VBD wanted) (S (VP (TO to) (VP (VB stay))))) (. .))");
        assert_eq!(l, vec!["e1 wanted  subject=She  clause=e2", "e2 stay [to]  subject=She"]);
        let l = run("(S (NP (PRP She)) (VP (VBD left) (SBAR (IN because) (S (NP (PRP it)) (VP (VBD rained))))) (. .))");
        assert_eq!(l, vec!["e1 left  subject=She  clause:because=e2", "e2 rained  subject=it"]);
    }

    #[test]
    fn copula_and_collapsed_labels() {
        let l = run("(S (NP (DT The) (NN report)) (VP (VBD was) (ADJP (RB too) (JJ long))) (. .))");
        assert_eq!(l, vec!["e1 was  subject=The report  complement=too long"]);
        let t = crate::tree::Node::Phrase("S::VP".into(), vec![crate::tree::Node::leaf("VB", "Go"), crate::tree::Node::leaf(".", ".")]);
        let e = events(&t, 0, &[]);
        assert_eq!(lines(&e), vec!["e1 Go"]);
    }

    #[test]
    fn time_phrases_and_linking_verbs() {
        let l = run("(S (NP (PRP She)) (VP (VBD left) (NP (NN yesterday))) (. .))");
        assert_eq!(l, vec!["e1 left  subject=She  time=yesterday"]);
        let l = run("(S (NP (PRP He)) (VP (VBD became) (NP (NN president)) (NP (JJ last) (NN year))) (. .))");
        assert_eq!(l, vec!["e1 became  subject=He  complement=president  time=last year"]);
        let l = run("(S (NP (PRP He)) (VP (VBD spent) (NP (DT the) (NN day)) (NP (JJ last) (NN week))) (. .))");
        assert_eq!(l, vec!["e1 spent  subject=He  object=the day  time=last week"]);
        let l = run("(S (NP (PRP It)) (VP (VBD seemed) (ADJP (JJ odd))) (. .))");
        assert_eq!(l, vec!["e1 seemed  subject=It  complement=odd"]);
        let l = run("(S (NP (PRP He)) (VP (VP (VBD got) (NP (DT the) (NN report))) (CC and) (VP (VBD got) (ADJP (JJ angry)))) (. .))");
        assert_eq!(l, vec!["e1 got  subject=He  object=the report", "e2 got  subject=He  complement=angry"]);
    }

    #[test]
    fn arguments_point_at_mentions() {
        let t = parse("(S (NP (NNP Alice)) (VP (VBD sent) (NP (NNP Bob)) (NP (NP (DT the) (NN report)) (PP (IN of) (NP (DT the) (NN year))))) (. .))").unwrap();
        let m = mentions(&t, 0);
        let e = events(&t, 0, &m);
        let obj = e[0].args.iter().find(|a| a.role == "object").unwrap();
        assert_eq!((obj.start, obj.end, obj.head), (3, 8, 4));
        assert_eq!(m[obj.mention.unwrap()].text, "the report");
    }

    #[test]
    fn gold_relations_follow_traces() {
        let raw = parse(
            "( (S (NP-SBJ-2 (PRP it)) (VP (VBD was) (VP (VBN sent) (NP (-NONE- *-2)) (PP (IN by) (NP-LGS (NNP Alice))))) (. .)) )",
        )
        .unwrap();
        // passive: "it" is the surface subject of "sent", the traced object
        // is "it" itself
        assert_eq!(gold_relations(&raw), vec![(Rel::Subject, 2, 0), (Rel::Object, 2, 0)]);
        let raw = parse(
            "( (S (NP-SBJ (NP (DT The) (NN man)) (SBAR (WHNP-1 (WP who)) (S (NP-SBJ (-NONE- *T*-1)) (VP (VBD called))))) (VP (VBD smiled)) (. .)) )",
        )
        .unwrap();
        assert_eq!(gold_relations(&raw), vec![(Rel::Subject, 3, 1), (Rel::Subject, 4, 1)]);
        // an unindexed empty subject (PRO) is skipped
        let raw = parse("( (S (NP-SBJ (PRP She)) (VP (VBD wanted) (S (NP-SBJ (-NONE- *)) (VP (TO to) (VP (VB stay))))) (. .)) )").unwrap();
        assert_eq!(gold_relations(&raw), vec![(Rel::Subject, 1, 0)]);
        // our side on the cleaned tree agrees
        let clean = crate::treebank::clean(&raw).unwrap();
        let e = events(&clean, 0, &[]);
        assert_eq!(relations(&e), vec![(Rel::Subject, 1, 0), (Rel::Subject, 3, 0)]);
    }
}
