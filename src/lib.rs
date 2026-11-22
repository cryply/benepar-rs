//! Constituency parsing with the benepar_en3 model, in Rust.
//!
//! ```no_run
//! use benepar_rs::Parser;
//! let mut parser = Parser::load(std::path::Path::new("benepar_en3")).unwrap();
//! for parse in parser.parse_text("The time for action is now.", 1, 1).unwrap() {
//!     println!("{}", parse.trees[0].1);
//! }
//! ```

pub mod chart;
pub mod model;
#[allow(dead_code)]
pub mod spm;
#[allow(dead_code)]
pub mod t5;
pub mod text;
pub mod tree;
pub mod treebank;

pub use model::Parser;
pub use text::Sentence;
pub use tree::Node;

use anyhow::Result;

/// One parsed sentence: its words, predicted POS tags and the k best trees,
/// best first, each with its score.
#[derive(Clone, Debug)]
pub struct Parse {
    pub words: Vec<String>,
    pub tags: Vec<String>,
    pub trees: Vec<(f32, Node)>,
}

impl Parse {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "words": self.words,
            "tags": self.tags,
            "trees": self.trees.iter().map(|(score, t)| serde_json::json!({
                "score": score,
                "bracketed": t.to_string(),
                "tree": t.to_json(),
            })).collect::<Vec<_>>(),
        })
    }
}

impl Parser {
    /// Parse one tokenized sentence. `k` is the number of trees wanted;
    /// `label_variants` is how many labels a span may try when k > 1
    /// (1 keeps the best label and varies only the bracketing).
    pub fn parse_sentence(&mut self, sent: &Sentence, k: usize, label_variants: usize) -> Result<Parse> {
        let out = self.parse(&sent.words, &sent.space_after)?;
        let leaves: Vec<Node> = sent
            .words
            .iter()
            .zip(&out.tags)
            .map(|(w, t)| Node::leaf(t, w))
            .collect();
        let k = k.max(1);
        // Several binarizations of one n-ary tree score the same and print
        // the same, so decode more derivations than trees wanted.
        let c = chart::Chart::decode(&out.scores, k * 4, label_variants, self.force_root);
        let mut trees: Vec<(f32, Node)> = Vec::new();
        for r in 0..c.len() {
            let t = c.tree(r, &self.labels, &leaves);
            if trees.iter().any(|(_, seen)| *seen == t) {
                continue;
            }
            trees.push((c.score(r), t));
            if trees.len() == k {
                break;
            }
        }
        Ok(Parse { words: sent.words.clone(), tags: out.tags, trees })
    }

    /// Split raw text into sentences and parse each one.
    pub fn parse_text(&mut self, text: &str, k: usize, label_variants: usize) -> Result<Vec<Parse>> {
        text::sentences(text)
            .iter()
            .map(|s| self.parse_sentence(s, k, label_variants))
            .collect()
    }
}
