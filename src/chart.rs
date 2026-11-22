// CKY decoding over labeled span scores, with k-best output.
//
// scores[i][j][l] is the score of label l for the span of words i..=j.
// Label 0 is the empty label: the span is only a binarization node. As in
// benepar, a tree is a binary bracketing of the sentence where every span
// gets its best label, and the tree score is the sum over its spans. With
// force_root the full-sentence span may not take the empty label.
//
// k-best: every span keeps its k best derivations. A derivation is a label
// choice plus, for spans longer than one word, a split point and a rank in
// each child. Candidates are combined exhaustively and truncated to k, which
// is cheap for sentence-length inputs.

use crate::tree::Node;

#[derive(Clone, Copy, Debug)]
struct Deriv {
    score: f32,
    label: usize,
    split: usize,
    left: usize,
    right: usize,
}

pub struct Chart {
    n: usize,
    cells: Vec<Vec<Deriv>>,
}

impl Chart {
    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.n + j
    }

    // label_k: how many labels each span may try. 1 keeps only the best
    // label per span, so alternative trees differ in bracketing, which is
    // what separates readings such as PP attachment.
    pub fn decode(scores: &[Vec<Vec<f32>>], k: usize, label_k: usize, force_root: bool) -> Self {
        let n = scores.len();
        let k = k.max(1);
        let mut chart = Chart {
            n,
            cells: vec![Vec::new(); n * n],
        };
        for len in 1..=n {
            for i in 0..=n - len {
                let j = i + len - 1;
                let s = &scores[i][j];
                let zero = s[0];
                let is_root = len == n && force_root;
                let mut labels: Vec<(f32, usize)> = s
                    .iter()
                    .enumerate()
                    .filter(|(l, _)| !(is_root && *l == 0))
                    .map(|(l, v)| (v - zero, l))
                    .collect();
                labels.sort_by(|a, b| b.0.total_cmp(&a.0));
                labels.truncate(label_k.max(1).min(k));

                let mut cand: Vec<Deriv> = Vec::new();
                if len == 1 {
                    for &(ls, l) in &labels {
                        cand.push(Deriv { score: ls, label: l, split: 0, left: 0, right: 0 });
                    }
                } else {
                    let mut kids: Vec<(f32, usize, usize, usize)> = Vec::new();
                    for m in i + 1..=j {
                        let lc = &chart.cells[chart.idx(i, m - 1)];
                        let rc = &chart.cells[chart.idx(m, j)];
                        for (a, ld) in lc.iter().enumerate() {
                            for (b, rd) in rc.iter().enumerate() {
                                kids.push((ld.score + rd.score, m, a, b));
                            }
                        }
                    }
                    kids.sort_by(|a, b| b.0.total_cmp(&a.0));
                    kids.truncate(k);
                    for &(ls, l) in &labels {
                        for &(ks, m, a, b) in &kids {
                            cand.push(Deriv { score: ls + ks, label: l, split: m, left: a, right: b });
                        }
                    }
                }
                cand.sort_by(|a, b| b.score.total_cmp(&a.score));
                cand.truncate(k);
                let at = chart.idx(i, j);
                chart.cells[at] = cand;
            }
        }
        chart
    }

    pub fn len(&self) -> usize {
        if self.n == 0 {
            0
        } else {
            self.cells[self.idx(0, self.n - 1)].len()
        }
    }

    pub fn score(&self, rank: usize) -> f32 {
        self.cells[self.idx(0, self.n - 1)][rank].score
    }

    // Tree for the rank-th best derivation. Empty-label nodes are flattened
    // into their parent, so the result is n-ary like the treebank.
    pub fn tree(&self, rank: usize, labels: &[String], leaves: &[Node]) -> Node {
        let mut kids = self.build(0, self.n - 1, rank, labels, leaves);
        if kids.len() == 1 {
            kids.pop().unwrap()
        } else {
            Node::Phrase(String::new(), kids)
        }
    }

    fn build(&self, i: usize, j: usize, rank: usize, labels: &[String], leaves: &[Node]) -> Vec<Node> {
        let d = self.cells[self.idx(i, j)][rank];
        let children = if i == j {
            vec![leaves[i].clone()]
        } else {
            let mut c = self.build(i, d.split - 1, d.left, labels, leaves);
            c.extend(self.build(d.split, j, d.right, labels, leaves));
            c
        };
        let label = &labels[d.label];
        if label.is_empty() {
            children
        } else {
            vec![Node::Phrase(label.clone(), children)]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaves(n: usize) -> Vec<Node> {
        (0..n).map(|i| Node::leaf("X", &format!("w{i}"))).collect()
    }

    #[test]
    fn picks_best_bracketing_and_root_label() {
        // labels: 0 = empty, 1 = A, 2 = B
        let labels = vec![String::new(), "A".into(), "B".into()];
        let mut s = vec![vec![vec![0.0f32; 3]; 3]; 3];
        s[0][1][1] = 5.0; // (A w0 w1)
        s[1][2][2] = 1.0; // weaker (B w1 w2)
        s[0][2][0] = 10.0; // empty is best for the root, but root is forced
        s[0][2][2] = 0.5;
        let c = Chart::decode(&s, 3, 3, true);
        let t = c.tree(0, &labels, &leaves(3));
        assert_eq!(t.to_string(), "(B (A (X w0) (X w1)) (X w2))");
        assert!(c.len() >= 2);
        assert!(c.score(0) >= c.score(1));
        let t1 = c.tree(1, &labels, &leaves(3));
        assert_ne!(t1.to_string(), t.to_string());
    }
}
