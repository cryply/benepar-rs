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

The full specification of the resolver, with what the current code
leaves out, is in [docs/pronouns.md](docs/pronouns.md). In short, the
rules: number agreement; names for he/she, descriptions for it;
gender from a small list of common first names, from the noun of a
description ("the widow", "my stepfather"), plus whatever earlier
pronouns in the same text clearly implied; a preference for subjects, for
names mentioned several times (what the text is about) and for mentions
close to the pronoun; a possessor ("Smith's room") counts for less and
the subject of a verb of speaking ("Holmes answered") for more; a name
after a preposition of place ("in Evanston") is not a person unless its
gender is known, and a name seen once after such a preposition, or twice
after to/from, and never with a title or a gender, is a place everywhere
in the text ("Stoke Moran", "Leatherhead"); the "it" of "it is my belief
that..." and "it was his custom to..." stands for no one and gets no
antecedent; forms of one name ("Smith",
"Alice Smith") are one candidate and share a gender, but a different title
("Miss Roylott" after "Dr. Roylott") or a known different gender makes
two people; reflexives bind in their own sentence. `--mentions`
also prints an entity index: each name or description with the sentences
it occurs in, counting sentences that reach it through a pronoun. This is
the hook for indexing notes by what they mention. It is a baseline, not a
coreference system: no world knowledge, no plural-to-group matching, no
"the company" = "Acme".

Every mention and candidate in `--mentions` output carries `span`, its
byte offsets in the input text. Possessives inside a longer noun phrase
("his support") are pronoun mentions of their own. A flat coordination
("John and Mary") is a plural mention, and its names are mentions marked
`conjunct`: they count for the entity index and for `--pronominalize`,
but are not offered as antecedents, since the rules cannot tell when one
conjunct is meant (offering them cost 0.9 F1 on GAP).

A pronoun's margin is how far its best antecedent scores ahead of the next
candidate that is not the same name ("Smith" and "Alice Smith" count as
one). `--canonical --min-margin X` leaves pronouns with a margin below X
without a bracket, or, with `--alternatives N`, gives them the candidates
within that margin instead of an answer: "he [Smith or Jones]".

### More pronouns

`--pronominalize` goes the other way: a repeated name or description
becomes a pronoun where the pronoun still points back to it.

```
$ benepar-rs --model ./benepar_en3 --pronominalize "Alice sent Bob the report. Bob read the report and told Alice that the report was too long. Alice's manager agreed with Bob."
Alice sent Bob the report .
He read it and told her that it was too long .
Her manager agreed with him .
```

A mention is a candidate when the same entity was mentioned in the same
or the previous sentence: the same name or part of it ("Smith" after
"Alice Smith"), or a definite description with the same head and no new
words ("the engine" after "the new engine"). The pronoun takes its gender
from the name list or from pronouns that already refer to the entity,
and its case from the position: subject, possessive (`Alice 's` becomes
`her`), otherwise object. Left as they are: names of unknown gender,
singular descriptions of people ("the teacher"), conjuncts of a
coordinated noun phrase, and objects whose entity is already named in
the same sentence (they would need a reflexive).

Every replacement is checked by resolving the rewritten text again with
the rules above. It is kept only if the new pronoun resolves to the same
entity, with at least `--min-margin` over the next candidate, and every
other pronoun keeps its antecedent. With `--json` the output is one
object with the rewritten lines and the replacements: position, pronoun,
the words replaced and the entity, which is the answer key for a reader
of the rewritten text.

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
them are right, at several margins, by distance to the chosen antecedent
and when only one candidate survived the hard filters, and how many gold
names occur only after the pronoun (the rules look back only).

On gap-development (2000 examples) with benepar_en3: restricted F1 63.3
overall (M 64.2, F 62.3), free F1 57.7 (precision 65.3, recall 51.6). The
margin separates right from wrong answers only weakly: precision goes from
65.0 with every prediction kept to 72.2 at margin 2.0, where 31% of the
examples keep a prediction. 353 of the 1799 gold names occur after the
pronoun.

### Does a language model read it better?

`examples/llm_test.rs` asks a chat model the GAP or Winograd question
("who does the marked pronoun refer to: A, B or neither") about the same
examples in several forms of the text and compares accuracy:

- `raw`: the text as it is
- `plain`: the text tokenized and joined back, as the other two see it
- `canonical`: every resolved pronoun followed by its antecedent in brackets
- `pronominalize`: repeated names turned into pronouns

```
$ export LLM_API_KEY=...            # OpenRouter by default, or set LLM_BASE_URL
$ cargo run --release --example llm_test -- --model MODEL --gap gap-development.tsv --limit 300
$ cargo run --release --example llm_test -- --model MODEL --wsc WSCollection.xml --limit 273
```

Any OpenAI-compatible chat endpoint works (`LLM_BASE_URL`, default
`https://openrouter.ai/api/v1`; for Ollama `http://HOST:11434/v1` with no
key). `--reasoning-effort low` keeps a thinking model short; its answer
is read from `content`, not from its reasoning. Requests go out at
`--rpm` per minute,
429 and 5xx answers are retried with backoff, and every reply is cached
in `--cache` (default `llm-cache/`), so a rerun is free and a run cut
short continues where it stopped. `--dry-run` prints the prompts of the
first examples instead. `--out FILE` writes one JSON line per example
and condition. The summary gives accuracy with a 95% Wilson interval,
prompt tokens as the endpoint reports them, and an exact McNemar test of
every condition against the first one. The Winograd Schema Collection
(https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WSCollection.xml)
holds 285 schemas; the first 273 are WSC273.

### A character's biography

`examples/story_test.rs` asks the model for every fact a story gives
about one character, then has the same model judge the list: which gold
facts it contains (recall) and which of its statements the original text
supports (precision), with the token cost of each form of the text. The
forms are `raw`, `plain`, `canonical`, `pronominalize`, `mentions` (only
the sentences that mention the character according to the entity index,
pronouns included) and `index` (no text: the character's descriptors and
the skeleton events it takes part in).

```
$ cargo run --release --example story_test -- --model MODEL --text speckled_band.txt \
      --character "Dr. Grimesby Roylott" --gold roylott.facts --out-dir story-out
```

The text is one paragraph per line; the gold file one fact per line.
`--out-dir` keeps every form of the text, the model's lists and the
judge's verdicts (missing facts, unsupported statements).

## Sentence skeleton

`--skeleton` reads the events of every sentence off the tree with head
rules: each verb with its subject, objects, prepositional and clause
arguments. The skeleton is kept apart from the words that fill it: with
`--json` every argument carries its word positions, its head word and the
index of the mention it is, so "sent(Alice; Bob, the report)" can be
stored next to the entities it links.

```
$ benepar-rs --model ./benepar_en3 --skeleton "Alice sent Bob the report. It was not approved by the board, which had been told to wait."
s0 e1 sent  subject=Alice  indirect=Bob  object=the report
s1 e1 approved [was] not passive  object=It  subject=the board
s1 e2 told [had been] passive  object=the board  clause=e3
s1 e3 wait [to]  subject=the board
```

Roles are grammatical, normalized for voice: in a passive the surface
subject is the `object` and the `by` phrase the `subject`. Auxiliaries,
modals and particles are listed in brackets, negation as `not`. Handled:
auxiliary chains, passives, verb coordination (one subject, two events),
prepositional arguments (`prep:into`), clause arguments (`clause:because`
points at the inner event), control ("told Alice to leave": Alice leaves;
"wanted to leave": the subject leaves), subject relatives and reduced
relatives (the noun is the subject of the inner event), copulas and
linking verbs with their `complement` ("became president", "got angry"),
bare time phrases as `time` ("left yesterday", "two weeks ago"). Not
handled: inversion, object relatives, ellipsis.
Nothing here knows what a verb means; roles are not semantic.

`--eval-skeleton FILE` scores the skeleton against the function tags of
the original treebank (`NP-SBJ`, traces and their coindexes): precision
and recall of (verb, subject head) and (verb, object head) pairs, both
from our parses and from the gold trees themselves, so that parse errors
and rule errors can be told apart. Takes the same files as `--eval`.

On section 23 (2416 sentences) with benepar_en3, (verb, subject) pairs
score P 72.4 / R 83.5 / F1 77.5 from our parses and 74.7 / 85.3 / 79.7
from the gold trees; (verb, object) pairs 82.6 / 92.0 / 87.1 and 84.9 /
93.6 / 89.0. Parsing costs about two points; the rest is the rules. The
subject precision gap is mostly by design: the rules fill in controlled
subjects that the treebank leaves empty.

## Entities

`--entities` lists who and what the text mentions: each entity with its
mentions (a name and its other forms, a description and its repeats, the
pronouns resolved to it), what the text says about it, and its roles in
the skeleton's events. With `--json` every mention carries its position
and byte span and every role its event, so the output can serve as an
index of the text for whoever keeps one; nothing here depends on any
store.

```
$ benepar-rs --model ./benepar_en3 --entities "Governor Angeloz met Alice Smith, the red-haired engineer from Munich, at the station. Smith had designed the new engine. She told him that the engine overheated, and he became worried."
#0 Governor Angeloz (him, he)  [title:Governor; complement:worried]  {subject of met, object of told, subject of became}
#1 Alice Smith (Smith, She)  [apposition:the red-haired engineer]  {object of met, subject of designed, subject of told}
#2 the red-haired engineer  [modifier:red-haired; head:engineer]
#3 Munich
#4 the station  [head:station]  {prep:at of met}
#5 the new engine (the engine)  [modifier:new; head:engine]  {object of designed, subject of overheated}
```

Descriptors come from four places: titles before a name ("Governor"),
modifiers and the head inside a description ("red-haired", "engineer"),
appositions ("Smith, the manager,"), and complements of copulas and
linking verbs ("became worried"). They are what a later implicit mention
("the governor", "the red-haired one") can be matched against. One such
match is done: a definite description whose head is a title ("the
Doctor", "the Governor") joins the one named entity that carries that
title, when there is exactly one.

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
