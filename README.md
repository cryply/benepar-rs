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

Every mention and candidate in `--mentions` output carries `span`, its
byte offsets in the input text. Possessives inside a longer noun phrase
("his support") are pronoun mentions of their own.

A pronoun's margin is how far its best antecedent scores ahead of the next
candidate that is not the same name ("Smith" and "Alice Smith" count as
one). `--canonical --min-margin X` leaves pronouns with a margin below X
without a bracket.

### Scoring on GAP

`--gap FILE` scores the pronoun rules on GAP (Webster et al., TACL 2018,
https://github.com/google-research-datasets/gap-coreference, Apache 2.0):
Wikipedia snippets with one pronoun and two names, each labeled as the
antecedent or not. The data is not part of this repository.

```
$ curl -LO https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv
$ benepar-rs --model ./benepar_en3 --gap gap-development.tsv --threads 8
```

It prints F1 for masculine and feminine pronouns, their ratio and the
overall F1, in two readings of the candidate list: `free` takes only the
best candidate, `restricted` the best candidate that is one of the two
names. It also prints how many free predictions are kept, and how many of
them are right, at several margins, and how many gold names occur only
after the pronoun (the rules look back only).

On gap-development (2000 examples) with benepar_en3: restricted F1 60.8
overall (M 60.6, F 60.9), free F1 53.4 (precision 62.2, recall 46.9). The
margin separates right from wrong answers only weakly: precision goes from
62.2 with every prediction kept to 67.7 at margin 2.0, where 24% of the
examples keep a prediction. 353 of the 1799 gold names occur after the
pronoun.

## Hyphenated compounds

`--compounds` lists the hyphenated words of every sentence with their parts
and the role of the left part, worked out from the head's tag; `--json`
output carries the same list under `compounds`.

```
$ benepar-rs --model ./benepar_en3 --compounds "The owner-controlled firm bought a time-consuming, 16-year-old, well-known plan."
{"text":"owner-controlled","left":"owner","head":"controlled","left_tag":"NN","head_tag":"VBN","pattern":"N-Ved","role":null,"reading":"X controlled by/with/in owner",...}
{"text":"time-consuming","left":"time","head":"consuming","pattern":"N-Ving","role":"object","reading":"X consuming time",...}
{"text":"16-year-old","left":"16-year","head":"old","pattern":"Num-N-Adj","role":"measure","reading":"X 16-year old",...}
{"text":"well-known","left":"well","head":"known","pattern":"Adv-Ved","role":"manner","reading":"X known well",...}
```

The right element is the head and its type fixes the role of the left one
(`src/compounds.rs`): the object for N-Ving, manner for Adv-Ved and
Adv-Ving, the adjective's complement for N-Adj (oil-rich), an attributive
phrase for Adj-N and Num-N (long-term, three-page), a measure phrase for
Num-N-Adj, coordination for chains of nouns (Rome-Berlin-Tokyo). Chains are
reported once, as the outer compound, with the inner one as the left part
(owner-operator-controlled). Collapsed phrases (state-of-the-art,
easy-to-use) get `pattern: "phrase"` and read as ordinary syntax without
the hyphens.

N-Ved is the one pattern the tree does not decide: the left noun can be
the agent (owner-controlled), an instrument (hand-made) or a place
(London-based), so `role` is left `null` there for a lexicon or a model to
fill in. Keyed by the two lemmas, one such entry covers owner-controlled
and owner-controlling alike.

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
