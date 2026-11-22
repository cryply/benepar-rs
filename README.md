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

## Mentions and pronouns

`--mentions` lists what the text talks about: names (runs of proper nouns),
descriptions (noun phrases headed by a common noun) and pronouns. Every
pronoun gets a ranked list of possible antecedents, not one answer, so the
ambiguity stays visible. `--canonical` prints the text with each pronoun
followed by its best antecedent in brackets:

```
$ benepar-rs --model ./benepar_en3 --canonical "Alice sent Bob the report. He read it and told her that it was too long."
Alice sent Bob the report .
He [Bob] read it [the report] and told her [Alice] that it [the report] was too long .
```

The rules: number agreement; names for he/she, descriptions for it;
gender from a small list of common first names, plus whatever earlier
pronouns in the same text already implied; a preference for subjects and
for recent mentions; reflexives bind in their own sentence. `--mentions`
also prints an entity index: each name or description with the sentences
it occurs in, counting sentences that reach it through a pronoun. This is
the hook for indexing notes by what they mention. It is a baseline, not a
coreference system: no world knowledge, no plural-to-group matching, no
"the company" = "Acme".

## Model files

Get `benepar_en3` (T5-small, 49 MB download) or `benepar_en3_large`
(T5-large, 716 MB download, 3 GB unpacked) from the benepar model index
(https://kitaev.com/benepar/index.xml) and unpack it. The directory must
contain `config.json`, `spiece.model` and `benepar_model.bin`. A
`benepar_model.bin.bz2` next to them is decompressed on first run. The
weights are not part of this repository.

`--model` can be replaced by the `BENEPAR_MODEL` environment variable.

## How it works

- Text is split into sentences and treebank-style words by simple rules
  (`src/text.rs`); the Python package uses spaCy for this step. Hyphenated
  words are split at the hyphen (`owner - controlled`, tagged `NN HYPH
  VBN`), as in the revised treebank the models were trained on; a word
  kept whole gets one leaf and usually a wrong tag. Words with a prefix
  such as `self-` or `non-`, numbers and URLs stay whole. `--keep-hyphens`
  turns the splitting off.
- Words are re-tokenized into T5 subwords. The tokenizer is built from
  `spiece.model` directly (`src/spm.rs`).
- T5 runs as encoder plus decoder; the decoder states at the last
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
$ benepar-rs --model ./benepar_en3 --eval /path/to/LDC2015T13/data/penntree/23 --threads 8
```

`--eval` takes a file or a directory of `.mrg`/`.tree` files, as shipped by
LDC. Trees are cleaned the way benepar's corpus scripts do it: empty
elements (`-NONE-`) and the constituents they leave empty are removed,
function tags and indices are stripped (`NP-SBJ-2` becomes `NP`). The
treebank itself is licensed by LDC and is not part of this repository.

On the revised WSJ test set (LDC2015T13, section 23, 2416 sentences), the
set the benepar authors report on, this port gets F1 95.28 against their
95.40 for benepar_en3, with exact match 52.36% and POS accuracy 97.80%,
about 0.05 s per sentence on 8 threads. The remaining 0.1 is within the
difference between this scorer and EVALB and the spacing guessed from
tokenized trees.

On `data/23.auto.clean` from the benepar repository (classic PTB
annotation, which the model was not trained on) it gets F1 93.82.

`benepar_en3_large` loads with the same code and gets F1 96.21 on the
same test set against the authors' 96.29, with exact match 56.46% and POS
accuracy 97.90%. It is about 9 times slower (0.45 s per sentence on 8
threads) and needs about 3 GB of memory.

## Build

```
cargo build --release
cargo test
```

CPU only for now. `--threads N` parses N sentences at once on separate
threads; the weights are shared. Matrix products already use all cores for
one sentence, so on a 2-core machine this gives about 20%, on a machine with
many cores much more. Output order is unchanged.

## Limits

- Only the benepar_en3 and benepar_en3_large layouts (T5) are supported.
- The rule-based tokenizer covers common English; unusual text may split
  differently from spaCy and then parse differently.
- Sentences up to 510 subword tokens.

## License

MIT. The original benepar code is Copyright (c) 2017-2018 Nikita Kitaev,
Victor Huang and Mitchell Stern, MIT License; src/t5.rs comes from the
candle project (MIT or Apache-2.0). Both license texts are in NOTICE. The
benepar weights are distributed by the benepar authors under their own
terms and are not part of this repository.
