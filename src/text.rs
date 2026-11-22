// Sentence splitting and word tokenization for raw English text.
//
// benepar expects treebank-style words plus a flag per word saying whether a
// space followed it in the original text. The Python package takes these from
// spaCy; this is a small rule-based replacement that follows the same
// conventions for the common cases: punctuation split off, clitics such as
// 's, n't, 're split from their host word, quotes normalized.

#[derive(Clone, Debug, PartialEq)]
pub struct Sentence {
    pub words: Vec<String>,
    pub space_after: Vec<bool>,
}

const CLITICS: [&str; 8] = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m", "'S"];

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

// Split one whitespace-free chunk into words.
fn split_chunk(chunk: &str) -> Vec<String> {
    let chars: Vec<char> = chunk.chars().collect();
    let mut lo = 0;
    let mut hi = chars.len();
    let mut prefix = Vec::new();
    let mut suffix = Vec::new();

    while lo < hi && (is_punct(chars[lo]) || chars[lo] == '\'') {
        prefix.push(chars[lo].to_string());
        lo += 1;
    }
    while hi > lo && is_punct(chars[hi - 1]) {
        hi -= 1;
        suffix.push(chars[hi].to_string());
    }
    // "..." stays one word
    if suffix.len() >= 3 && suffix.iter().all(|s| s == ".") {
        let n = suffix.len();
        suffix = vec![".".repeat(n)];
    }

    let mut core: String = chars[lo..hi].iter().collect();
    let mut tail = Vec::new();
    let lower = core.to_lowercase();
    for c in CLITICS {
        let cl = c.to_lowercase();
        if lower.len() > cl.len() && lower.ends_with(&cl) {
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

    let mut out = prefix;
    if !core.is_empty() {
        out.push(core);
    }
    out.extend(tail);
    out.extend(suffix.into_iter().rev());
    out
}

fn ends_sentence(word: &str) -> bool {
    matches!(word, "." | "!" | "?" | "...") || word.ends_with("?!")
}

pub fn sentences(text: &str) -> Vec<Sentence> {
    let mut out = Vec::new();
    let mut cur = Sentence { words: Vec::new(), space_after: Vec::new() };
    let chunks: Vec<&str> = text.split_whitespace().collect();
    for (ci, chunk) in chunks.iter().enumerate() {
        let words = split_chunk(chunk);
        let last = words.len().saturating_sub(1);
        for (wi, w) in words.into_iter().enumerate() {
            let space = wi == last;
            let end = ends_sentence(&w);
            cur.words.push(normalize_word(&w));
            cur.space_after.push(space);
            let next_starts_upper = chunks
                .get(ci + 1)
                .and_then(|c| c.chars().find(|ch| ch.is_alphanumeric() || *ch == '"'))
                .map(|ch| ch.is_uppercase() || ch == '"' || ch.is_numeric())
                .unwrap_or(true);
            if end && wi == last && next_starts_upper {
                if let Some(s) = cur.space_after.last_mut() {
                    *s = false;
                }
                out.push(std::mem::replace(
                    &mut cur,
                    Sentence { words: Vec::new(), space_after: Vec::new() },
                ));
            }
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
        let s = sentences("I don't know (really), said Bob.");
        assert_eq!(
            s[0].words,
            ["I", "do", "n't", "know", "(", "really", ")", ",", "said", "Bob", "."]
        );
    }
}
