# benepar-rs

A Rust port of the Berkeley Neural Parser (benepar) by Nikita Kitaev:
constituency parsing for English with the benepar_en3 model, without Python
or PyTorch.

The original is https://github.com/nikitakit/self-attentive-parser (MIT),
described in "Constituency Parsing with a Self-Attentive Encoder" (Kitaev
and Klein, ACL 2018) and "Multilingual Constituency Parsing with
Self-Attention and Pre-Training" (Kitaev, Cao and Klein, ACL 2019). This
port covers inference only: the model is loaded from the files the benepar
authors publish, and the network and the decoder are reimplemented in Rust.
Training is not ported.

```
$ benepar-rs --model ./benepar_en3 "The time for action is now."
(S (NP (NP (DT The) (NN time)) (PP (IN for) (NP (NN action)))) (VP (VBZ is) (ADVP (RB now))) (. .))
```

This is the same tree the Python package prints for this sentence in its
README.

## k-best trees

A sentence can have several readings. `-k N` prints the N best trees, best
first; `--scores` adds the tree score. By default the alternatives differ in
bracketing (attachment); `--label-variants M` also lets each span try its M
best labels.

```
$ benepar-rs --model ./benepar_en3 -k 3 --scores "I saw the man with the telescope."
```

## JSON output and library use

`--json` prints one JSON object per sentence with the words, the predicted
POS tags and the trees, each as a bracketed string and as nested
`{"label", "children"}` / `{"tag", "word"}` objects.

The crate is also a library:

```rust
use benepar_rs::Parser;

let mut parser = Parser::load(std::path::Path::new("benepar_en3"))?;
for parse in parser.parse_text("The time for action is now.", 1, 1)? {
    println!("{}", parse.trees[0].1);
}
```

`parse_sentence` takes an already tokenized `Sentence`; `Parse::to_json`
gives the same object the CLI prints.

## Model files

Get `benepar_en3` from the benepar model index
(https://kitaev.com/benepar/index.xml) and unpack it. The directory must
contain `config.json`, `spiece.model` and `benepar_model.bin`. A
`benepar_model.bin.bz2` next to them is decompressed on first run. The
weights are not part of this repository.

`--model` can be replaced by the `BENEPAR_MODEL` environment variable.

## How it works

- Text is split into sentences and treebank-style words by simple rules
  (`src/text.rs`); the Python package uses spaCy for this step.
- Words are re-tokenized into T5 subwords. The tokenizer is built from
  `spiece.model` directly (`src/spm.rs`).
- T5-small runs as encoder plus decoder; the decoder states at the last
  subword of each word are the word features (`src/t5.rs`, from candle).
- A two-layer partitioned transformer keeps content and position features
  apart, then an MLP scores every label for every span and every POS tag
  (`src/model.rs`).
- CKY finds the best tree, or the k best (`src/chart.rs`).

## Accuracy

`--eval FILE` scores the parser on a treebank file with one bracketed tree
per line and prints EVALB-style numbers (TOP ignored, punctuation not
counted, ADVP and PRT merged). The words are taken from the gold trees, so
this measures the model port, not the tokenizer.

```
$ benepar-rs --model ./benepar_en3 --eval 23.auto.clean
```

On WSJ section 23 as shipped in the benepar repository
(`data/23.auto.clean`, 2416 sentences, classic PTB tokenization and
annotation) this port gets F1 93.82, exact match 40.89%, POS accuracy
95.80%, at about 0.09 s per sentence on two CPU cores. The benepar authors
report F1 95.40 for benepar_en3 on the revised WSJ test set (LDC2015T13),
which uses different tokenization and annotation guidelines, so the two
numbers are not directly comparable; the gap is in line with that
difference.

## Build

```
cargo build --release
cargo test
```

CPU only for now.

## Limits

- Only the benepar_en3 model layout (t5-small) is supported.
- The rule-based tokenizer covers common English; unusual text may split
  differently from spaCy and then parse differently.
- Sentences up to 510 subword tokens.

## License

MIT. The original benepar code is Copyright (c) 2017-2018 Nikita Kitaev,
Victor Huang and Mitchell Stern, MIT License; src/t5.rs comes from the
candle project (MIT or Apache-2.0). Both license texts are in NOTICE. The
benepar_en3 weights are distributed by the benepar authors under their own
terms and are not part of this repository.
