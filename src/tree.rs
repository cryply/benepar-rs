// Parse trees and their bracketed (PTB) string form.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    // (tag, word)
    Leaf(String, String),
    // Label may be a collapsed unary chain such as "S::VP"; it is expanded
    // when printed. An empty label means no node of its own.
    Phrase(String, Vec<Node>),
}

impl Node {
    pub fn leaf(tag: &str, word: &str) -> Self {
        Node::Leaf(tag.to_string(), word.to_string())
    }

    // Labels of this node, outermost first.
    pub fn labels(&self) -> Vec<&str> {
        match self {
            Node::Leaf(..) => Vec::new(),
            Node::Phrase(l, _) => l.split("::").filter(|s| !s.is_empty()).collect(),
        }
    }

    // Constituents in pre-order as (labels, first word, end word exclusive).
    pub fn spans(&self) -> Vec<(String, usize, usize)> {
        let mut out = Vec::new();
        self.collect(0, &mut out);
        out
    }

    fn collect(&self, start: usize, out: &mut Vec<(String, usize, usize)>) -> usize {
        match self {
            Node::Leaf(..) => start + 1,
            Node::Phrase(l, kids) => {
                let at = out.len();
                if !l.is_empty() {
                    out.push((l.clone(), start, start));
                }
                let mut end = start;
                for k in kids {
                    end = k.collect(end, out);
                }
                if !l.is_empty() {
                    out[at].2 = end;
                }
                end
            }
        }
    }
}

impl Node {
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Node::Leaf(tag, word) => serde_json::json!({"tag": tag, "word": word}),
            Node::Phrase(_, kids) => {
                let kids: Vec<_> = kids.iter().map(Node::to_json).collect();
                let labels = self.labels();
                // Expand a collapsed unary chain into nested nodes.
                let mut node = serde_json::json!({"label": labels.last().copied().unwrap_or(""), "children": kids});
                for l in labels.iter().rev().skip(1) {
                    node = serde_json::json!({"label": l, "children": [node]});
                }
                node
            }
        }
    }
}

fn escape(word: &str) -> String {
    match word {
        "(" => "-LRB-".into(),
        ")" => "-RRB-".into(),
        "[" => "-LSB-".into(),
        "]" => "-RSB-".into(),
        "{" => "-LCB-".into(),
        "}" => "-RCB-".into(),
        _ => word.to_string(),
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Leaf(tag, word) => write!(f, "({} {})", escape(tag), escape(word)),
            Node::Phrase(label, kids) => {
                let labels: Vec<&str> = label.split("::").filter(|s| !s.is_empty()).collect();
                for l in &labels {
                    write!(f, "({l} ")?;
                }
                for (i, k) in kids.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{k}")?;
                }
                for _ in &labels {
                    write!(f, ")")?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_expands_unary_chain() {
        let t = Node::Phrase("S::VP".into(), vec![Node::leaf("VB", "Go")]);
        assert_eq!(
            t.to_json().to_string(),
            r#"{"children":[{"children":[{"tag":"VB","word":"Go"}],"label":"VP"}],"label":"S"}"#
        );
    }

    #[test]
    fn unary_chain_is_expanded() {
        let t = Node::Phrase(
            "S::VP".into(),
            vec![Node::leaf("VB", "Go"), Node::leaf(".", ".")],
        );
        assert_eq!(t.to_string(), "(S (VP (VB Go) (. .)))");
        assert_eq!(t.spans(), vec![("S::VP".to_string(), 0, 2)]);
    }
}
