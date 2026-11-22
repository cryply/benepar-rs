// Sentence splitting and word tokenization for raw English text.
//
// benepar expects treebank-style words plus a flag per word saying whether a
// space followed it in the original text. The Python package takes these from
// spaCy; this is a rule-based replacement that follows the same conventions
// for the common cases: punctuation split off, clitics such as 's and n't
// split from their host, abbreviations and numbers kept whole, quotes
// normalized.

#[derive(Clone, Debug, PartialEq)]
pub struct Sentence {
    pub words: Vec<String>,
    pub space_after: Vec<bool>,
}

const CLITICS: [&str; 7] = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"];

// Words that end in a period which is part of the word, not a sentence end.
const ABBREVIATIONS: [&str; 42] = [
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "mt.", "vs.",
    "etc.", "e.g.", "i.e.", "cf.", "inc.", "ltd.", "co.", "corp.", "no.",
    "u.s.", "u.k.", "u.n.", "a.m.", "p.m.", "jan.", "feb.", "mar.", "apr.",
    "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.", "gen.",
    "sen.", "rep.", "gov.", "lt.", "col.",
];

// Words spaCy splits in two.
const SPLITS: [(&str, &str, &str); 4] = [
    ("cannot", "can", "not"),
    ("gonna", "gon", "na"),
    ("wanna", "wan", "na"),
    ("gotta", "got", "ta"),
];

// Same mapping benepar applies to spaCy tokens before parsing.
pub fn normalize_word(w: &str) -> String {
    match w {
        "`" | "\u{2018}" | "\u{2019}" | "\u{2039}" | "\u{203a}" => "'".into(),
        "\u{ab}" | "\u{bb}" | "\u{201c}" | "\u{201d}" | "\u{201e}" => "\"".into(),
        "\u{2014}" => "--".into(),
        _ => w.to_string(),
    }
}

fn is_punct(c: char) -> bool {
    !c.is_alphanumeric() && c != '\'' && c != '\u{2019}' && c != '-' && c != '_'
}

fn is_abbreviation(w: &str) -> bool {
    let l = w.to_lowercase();
    if ABBREVIATIONS.contains(&l.as_str()) {
        return true;
    }
    // Single capital letter with a period ("J. Smith"), or letters separated
    // by periods ("U.S.A.").
    let mut chars = l.chars();
    if let (Some(a), Some('.'), None) = (chars.next(), chars.next(), chars.next()) {
        return a.is_alphabetic();
    }
    l.ends_with('.')
        && l.len() > 2
        && l.chars().any(|c| c.is_alphabetic())
        && l.split('.').all(|p| p.len() <= 1 && p.chars().all(|c| c.is_alphabetic()))
}

// Numbers like 190.58, 1,000, 3:30, 1/2 stay one word.
fn is_number(w: &str) -> bool {
    let mut digits = 0;
    for c in w.chars() {
        if c.is_ascii_digit() {
            digits += 1;
        } else if !matches!(c, '.' | ',' | ':' | '/' | '-') {
            return false;
        }
    }
    digits > 0 && w.chars().next().map_or(false, |c| c.is_ascii_digit())
}

fn is_url_or_email(w: &str) -> bool {
    w.contains("://") || w.starts_with("www.") || (w.contains('@') && w.contains('.'))
}

// Split one whitespace-free chunk into words.
fn split_chunk(chunk: &str) -> Vec<String> {
    if is_url_or_email(chunk) {
        // keep the address whole, but a sentence-final mark still splits off
        let trimmed = chunk.trim_end_matches(|c| matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | ')'));
        let mut out = vec![trimmed.to_string()];
        out.extend(chunk[trimmed.len()..].chars().map(|c| c.to_string()));
        return out;
    }
    let chars: Vec<char> = chunk.chars().collect();
    let mut lo = 0;
    let mut hi = chars.len();
    let mut prefix = Vec::new();
    let mut suffix = Vec::new();

    while lo < hi && (is_punct(chars[lo]) || chars[lo] == '\'') {
        prefix.push(chars[lo].to_string());
        lo += 1;
    }
    loop {
        let core: String = chars[lo..hi].iter().collect();
        if hi > lo && is_punct(chars[hi - 1]) {
            // A trailing period stays on abbreviations and on numbers that
            // are not the end of the chunk ("3.5" but not "3.").
            if chars[hi - 1] == '.' && is_abbreviation(&core) {
                break;
            }
            hi -= 1;
            suffix.push(chars[hi].to_string());
        } else {
            break;
        }
    }
    // "..." stays one word
    if suffix.len() >= 3 && suffix.iter().all(|s| s == ".") {
        let n = suffix.len();
        suffix = vec![".".repeat(n)];
    }

    let mut core: String = chars[lo..hi].iter().collect();
    let mut tail = Vec::new();
    if !is_number(&core) {
        let lower = core.to_lowercase();
        for c in CLITICS {
            if lower.len() > c.len() && lower.ends_with(c) {
                let cut = core.len() - c.len();
                if core.is_char_boundary(cut) {
                    tail.push(core[cut..].to_string());
                    core.truncate(cut);
                }
                break;
            }
        }
        if core.ends_with('\'') && core.len() > 1 {
            core.pop();
            tail.insert(0, "'".into());
        }
    }

    let mut out = prefix;
    if !core.is_empty() {
        let lower = core.to_lowercase();
        match SPLITS.iter().find(|(w, _, _)| *w == lower) {
            Some((_, a, _)) => {
                out.push(core[..a.len()].to_string());
                out.push(core[a.len()..].to_string());
            }
            None => out.push(core),
        }
    }
    out.extend(tail);
    out.extend(suffix.into_iter().rev());
    out
}

fn is_closer(w: &str) -> bool {
    matches!(w, "\"" | "'" | ")" | "]" | "}" | "''")
}

fn is_terminal(w: &str) -> bool {
    matches!(w, "." | "!" | "?" | "...") || w.ends_with("?!") || w.ends_with("!?")
}

// Does this chunk's word list end a sentence: a terminal mark, possibly
// followed by closing quotes or brackets.
fn chunk_ends_sentence(words: &[String]) -> bool {
    let mut i = words.len();
    while i > 0 && is_closer(&words[i - 1]) {
        i -= 1;
    }
    i > 0 && is_terminal(&words[i - 1]) && !is_abbreviation(&words[i - 1])
}

pub fn sentences(text: &str) -> Vec<Sentence> {
    let mut out = Vec::new();
    let mut cur = Sentence { words: Vec::new(), space_after: Vec::new() };
    let chunks: Vec<&str> = text.split_whitespace().collect();
    for (ci, chunk) in chunks.iter().enumerate() {
        let words = split_chunk(chunk);
        let last = words.len().saturating_sub(1);
        let ends = chunk_ends_sentence(&words);
        for (wi, w) in words.into_iter().enumerate() {
            cur.words.push(normalize_word(&w));
            cur.space_after.push(wi == last);
        }
        let next_starts_sentence = chunks
            .get(ci + 1)
            .and_then(|c| c.chars().find(|ch| ch.is_alphanumeric() || *ch == '"'))
            .map(|ch| ch.is_uppercase() || ch == '"' || ch.is_numeric())
            .unwrap_or(true);
        if ends && next_starts_sentence {
            if let Some(s) = cur.space_after.last_mut() {
                *s = false;
            }
            out.push(std::mem::replace(
                &mut cur,
                Sentence { words: Vec::new(), space_after: Vec::new() },
            ));
        }
    }
    if !cur.words.is_empty() {
        if let Some(s) = cur.space_after.last_mut() {
            *s = false;
        }
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<Vec<String>> {
        sentences(text).into_iter().map(|s| s.words).collect()
    }

    #[test]
    fn readme_example() {
        let s = sentences("The time for action is now. It's never too late to do something.");
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].words, ["The", "time", "for", "action", "is", "now", "."]);
        assert_eq!(s[0].space_after, [true, true, true, true, true, false, false]);
        assert_eq!(
            s[1].words,
            ["It", "'s", "never", "too", "late", "to", "do", "something", "."]
        );
        assert_eq!(s[1].space_after[0], false);
    }

    #[test]
    fn punctuation_and_negation() {
        assert_eq!(
            words("I don't know (really), said Bob.")[0],
            ["I", "do", "n't", "know", "(", "really", ")", ",", "said", "Bob", "."]
        );
    }

    #[test]
    fn abbreviations_do_not_end_sentences() {
        let w = words("Dr. Smith met Mr. Jones in the U.S. last year. They talked.");
        assert_eq!(w.len(), 2);
        assert_eq!(
            w[0],
            ["Dr.", "Smith", "met", "Mr.", "Jones", "in", "the", "U.S.", "last", "year", "."]
        );
        assert_eq!(words("See J. Smith et al.")[0], ["See", "J.", "Smith", "et", "al", "."]);
    }

    #[test]
    fn numbers_stay_whole() {
        assert_eq!(
            words("It rose 190.58 points, or 1,000 dollars, at 3:30.")[0],
            ["It", "rose", "190.58", "points", ",", "or", "1,000", "dollars", ",", "at", "3:30", "."]
        );
        assert_eq!(words("Pay $5 or 10% now.")[0], ["Pay", "$", "5", "or", "10", "%", "now", "."]);
    }

    #[test]
    fn quotes_and_closers() {
        let w = words("He said \"go now.\" Then he left (quickly). Done!");
        assert_eq!(w.len(), 3);
        assert_eq!(w[0], ["He", "said", "\"", "go", "now", ".", "\""]);
        assert_eq!(w[1], ["Then", "he", "left", "(", "quickly", ")", "."]);
        assert_eq!(w[2], ["Done", "!"]);
    }

    #[test]
    fn urls_hyphens_and_splits() {
        assert_eq!(
            words("Visit https://example.com/a.b or mail bob@example.com.")[0],
            ["Visit", "https://example.com/a.b", "or", "mail", "bob@example.com", "."]
        );
        assert_eq!(
            words("A well-known fact: you cannot win.")[0],
            ["A", "well-known", "fact", ":", "you", "can", "not", "win", "."]
        );
        assert_eq!(words("Wait... really?")[0], ["Wait", "...", "really", "?"]);
        assert_eq!(words("Wait... Really?").len(), 2);
    }

    #[test]
    fn lowercase_after_period_does_not_split() {
        assert_eq!(words("v1. is out. see it.").len(), 1);
    }
}
