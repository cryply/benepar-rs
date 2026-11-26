# Pronoun resolution from constituency trees

This document is the specification of the pronoun resolver in
`src/mentions.rs`. The code is meant to follow it; where it does not, the
code is wrong. Section 10 lists what the current code does not do yet.

The input is one document: a list of sentences, each a Penn Treebank
style tree as benepar produces it. Benepar trees carry the 36 word tags
and the phrase labels of the treebank, but not its function tags (-SBJ,
-TMP, -PRD, ...), traces, or coindexes. Everything below is derived from
labels, tags and tree shape alone.

The output is, for every pronoun, a ranked list of antecedent candidates
with scores, or no list at all when the pronoun has no antecedent in the
text (expletive it, first and second person outside direct speech).

## 1. What a pronoun is here

A pronoun is a word with tag PRP or PRP$, a reflexive (PRP ending in
-self/-selves), or a demonstrative used on its own (DT alone under NP:
"this", "that", "these", "those"). Relative pronouns (WP, WDT, WP$ under
WHNP) are handled structurally, see 2.4; they never enter the candidate
search.

Each pronoun has these features:

| feature | values | source |
|---|---|---|
| person | 1, 2, 3 | the word |
| number | singular, plural, unknown | the word; "they" is unknown, see 1.1 |
| gender | male, female, neuter, unknown | the word |
| animate | yes, no, unknown | he/she yes, it no, they unknown |
| case | subject, object, possessive, reflexive | the word and the tag |

Word table (lowercased):

| words | person | number | gender | animate |
|---|---|---|---|---|
| i, me, my, mine, myself | 1 | sg | unknown | yes |
| we, us, our, ours, ourselves | 1 | pl | unknown | yes |
| you, your, yours, yourself, yourselves | 2 | unknown | unknown | yes |
| he, him, his, himself | 3 | sg | male | yes |
| she, her, hers, herself | 3 | sg | female | yes |
| it, its, itself | 3 | sg | neuter | no |
| they, them, their, theirs, themselves | 3 | unknown | unknown | unknown |
| this, that | 3 | sg | neuter | no |
| these, those | 3 | pl | neuter | no |
| one, oneself | 3 | sg | unknown | yes |

### 1.1 Singular they

"they" matches plural antecedents first. It also matches a singular
animate antecedent whose gender is unknown (a name not in the gender
list, "the student", "someone", "a person"), and a singular antecedent
that an earlier "they" in the text already resolved to. A singular
antecedent of known gender ("Mary", "the widow") is not a candidate for
"they" unless the text has already used "they" for it. A plural
antecedent and a singular one of unknown gender tie on agreement; the
rest of the score decides.

### 1.2 Reflexives

A reflexive binds inside its own clause (the nearest S or SINV above
it): the antecedent is the subject of that clause. If that subject is a
pronoun, the reflexive takes the pronoun's antecedent. Nothing outside
the clause is a candidate.

### 1.3 Possessives

"his", "her" as PRP$ inside an NP ("his hat") are mentions of their own;
the NP they open is a separate mention headed by its noun. The same
holds for a possessor NP with POS ("Smith's hat"): the possessor is a
mention, the whole NP is another, headed by "hat".

### 1.4 First and second person

Outside direct speech, "I" is the narrator and "you" the reader; they
get no antecedent in the text. Inside direct speech they refer to the
speaker and the addressee, see 6.

## 2. What the tree gives

### 2.1 Mentions

A mention is a maximal NP that contains no NP of its own, with these
exceptions: a possessor NP (ending in POS) inside it does not count, and
a coordination of NPs yields one plural mention for the whole and one
mention per conjunct. A run of NNP outside any NP is a mention too.

Each mention records: the sentence, word span, text, head word (the last
noun of the core, before any PP or clause), kind (name: all NNP/NNPS;
pronoun; nominal: everything else), number (NNS/NNPS or coordination:
plural; "the N" with NN: singular; mass and unknown: unknown), and the
flags below.

### 2.2 Grammatical role

Role comes from the position of the mention under the nearest clause:

- subject: the first NP child of an S, SINV or SQ, or the NP after the
  VP in an inverted clause ("said he"). A subject NP that is itself a
  coordination gives the subject role to the whole, not to the conjuncts.
- object: an NP child of a VP after the verb.
- oblique: the NP child of a PP.
- possessor: an NP ending in POS inside another NP.
- predicate: the NP after a copula (be, become, seem, remain, appear) in
  a VP, "Smith is the manager".
- apposition: an NP child of an NP that follows another NP child and a
  comma, "Smith, the manager,".

A predicate or apposition NP is not an antecedent of its own; it is a
description of the subject (or of the NP it is apposed to) and is
recorded as such.

### 2.3 Place

An NP right after a preposition of place (in, at, near, into, onto,
across, through, within, outside, inside, around, throughout, toward,
towards) is marked locative. An NP after "to" or "from" is marked
motion. A name that is locative at least once, or motion at least twice,
and never carries a title or a gendered first name, is a place in every
occurrence: it is not a candidate for he, she, they, and it is never
taught a gender.

### 2.4 Relative clauses

A WHNP (who, whom, whose, which, that as WDT) under an SBAR that is a
child of an NP refers to that NP's head: "the man who left" binds "who"
to "the man". This is read off the tree and is not scored. A WHNP under
an SBAR that is a child of a VP or S (free relative, "what he said",
"whoever comes") gets no antecedent.

### 2.5 Clauses and verb phrases as antecedents

"it", "this", "that" and "which" after a comma can refer to a clause or
to an event rather than to an NP: "He left early. This surprised
everyone." The candidates for a neuter pronoun therefore include, besides
NPs, the previous sentence as a whole and the VP of the previous clause.
They are scored low and win only when no neuter NP fits, see 4.3.

### 2.6 Expletive it

"it" is not a pronoun when it is the subject of a clause that has a
clause complement: an S or SBAR child of the clause itself, of its VP, or
of an ADJP in its VP. This covers "it is my belief that...", "it was his
custom to smoke", "it became clear to me that...", "made it impossible
for me to tell", "it seems that...". Nor is it a pronoun when it is the
subject of a copula whose predicate is a time noun, a number or a word
of weather or daylight ("it was late", "it was a wild night", "it was
seven o'clock", "it is a little cold"), or the subject of a raising verb
("it seems", "it appears", "as it happens", "it turned out").

### 2.7 Direct speech

A quoted span is the text between a pair of quote tags (`` `` `` and
`''`, or straight quotes). The speaker of a quote is the subject of the
speech verb (said, answered, asked, cried, remarked, replied, continued,
...) in the clause the quote is attached to: "..." said Holmes; Holmes
said "..."; "...," he answered, "...". A quote with no attribution takes
the speaker from the previous quote by alternation when exactly two
people have spoken in the last few turns, and otherwise has no known
speaker. The addressee is the other participant when known.

Inside a quote, "I/me/my" refers to the speaker, "you/your" to the
addressee, "we" to the speaker plus others. Outside quotes, first and
second person have no antecedent. A quote is also a boundary for
salience: a mention inside a quote is less prominent for a pronoun
outside it, and the other way round, because the speaker and the
narrator keep separate threads.

## 3. Candidate generation

For a third person pronoun, the candidates are the mentions before it in
the text, nearest first, back to the beginning of the document, that
pass the hard constraints of section 4. Pronouns are candidates too,
but a pronoun candidate stands for its own antecedent: the chain is
followed to the first non-pronoun, which is the candidate that is
scored, and the chain only contributes the pronoun's position and role
(section 5.2).

Mentions in the same sentence after the pronoun are candidates only for
a pronoun in a fronted clause ("When he arrived, Smith sat down"): the
pronoun is inside an SBAR or PP that precedes the main clause's subject,
and the candidate is that subject. No other cataphora is allowed.

A candidate set is capped at the five best after scoring, and a name is
listed once: "Smith" after "Alice Smith" is the same candidate, see 7.

## 4. Hard constraints

A candidate that fails any of these is dropped.

### 4.1 Agreement

Number: a singular pronoun needs a singular or unknown antecedent, a
plural pronoun a plural or unknown one. A coordination of two singulars
is plural. "they": see 1.1.

Gender: he needs male or unknown, she female or unknown. A known gender
that differs is a mismatch. Names get a gender from a list of first
names, from a title (Mr., Mrs., Miss, Ms., Sir, Lady, Lord), from the
head noun of a description apposed to the name ("Smith, the widow"), and
from earlier clear resolutions in the same text (section 7). Nominals get
a gender from their head noun (widow, stepfather, girl, Mrs. ...).

Animacy: he, she need animate or unknown; it needs inanimate or unknown,
and never a time ("the morning", "some time").
A name of a person is animate; a place is inanimate; a nominal is animate
when its head is a person noun (man, doctor, friend, -er/-or/-ist
agent nouns, kinship terms, titles) and inanimate when its head is a
known thing or event noun; otherwise unknown.

### 4.2 Binding

A non-reflexive pronoun cannot corefer with a mention that is the
subject of the same minimal clause: in "Smith saw him", "him" is not
Smith. A possessive is exempt: "Smith saw his hat" allows Smith. A
pronoun inside a PP or an embedded clause of the same sentence is
exempt as well: "Smith said that he left" allows Smith.

### 4.3 Neuter pronouns and events

For it/this/that/these/those the candidates are inanimate or unknown
NPs, plus the clause candidates of 2.5. An NP wins over a clause when it
is in the same or the previous sentence; a clause wins when the pronoun
is the subject of a copula with an adjective or abstract noun predicate
("this was strange", "that is the question") or the object of a verb of
saying or thinking ("he said that", "I know it").

## 5. Scoring

Each surviving candidate gets a score; the highest wins. The weights
below are the current ones; they are tuned on GAP and on the story test
and should be read as an ordering, not as calibrated probabilities.

### 5.1 Role

| role of the candidate | points |
|---|---|
| subject | +0.5 |
| speaker of a quote (subject of a speech verb) | +0.5 more |
| object | 0 |
| oblique, for a neuter pronoun | -0.5 |
| oblique, for he/she | 0 |
| possessor | -0.5 |
| predicate, apposition | not a candidate |

### 5.2 Recency and chains

Distance is counted in sentences: -0.5 per sentence between the
candidate and the pronoun. Inside the same sentence, a candidate closer
to the pronoun scores higher (+0.25, plus up to +0.5 by relative
position).

When a candidate was last referred to by a pronoun that itself resolved
clearly (section 8), the distance is counted from that pronoun, and the
role bonus is that pronoun's role. This keeps the subject of a run of
"he ... he ... his" sentences the preferred antecedent of the next "he"
even when another name appeared in between as an object or possessor.

### 5.3 Kind and gender fit

| candidate | points |
|---|---|
| name, animate pronoun | +2 |
| name with the pronoun's gender | +2 more |
| nominal with a gendered head of the pronoun's gender | +1 |
| nominal with a person head, gender unknown | +0.5 |
| nominal, neuter pronoun | +1.5 |
| name, neuter pronoun, gender unknown | +0.5 |

### 5.4 Frequency

A name mentioned n times before the pronoun gets +1 per mention beyond
the first, capped at +4, for animate pronouns only. This is what makes
the protagonist the default antecedent.

### 5.5 New entity

A nominal with an indefinite determiner (a, an, some, another) or no
determiner, in subject position in the sentence right before the
pronoun, introduces a new participant: +1.5. "A young lady has arrived.
She is waiting" resolves to the lady, not to a name from two sentences
earlier.

### 5.6 Repetition in a sentence

A pronoun word that already occurred in the same sentence and resolved
to a candidate gives that candidate +1 for the next occurrence of the
same word: a second "it" in one sentence usually keeps its referent.

### 5.7 Parallelism

A candidate with the same role as the pronoun (subject for a subject
pronoun, object for an object pronoun) in the previous clause gets
+0.25.

## 6. Direct speech, applied

Within a quote, candidates for third person pronouns are, in order of
preference: mentions inside the same quote, mentions in the narration
right before the quote, mentions in earlier quotes by the same speaker.
Mentions inside other speakers' quotes are candidates with -0.5.

The speaker of the quote is never a candidate for a third person pronoun
inside that quote ("I" covers that), and the addressee is not a
candidate for he/she inside the quote either when the addressee is
known.

## 7. Names and gender learning

Two name mentions are the same name when they share a capitalised word
that is not a title, and nothing says they are two people: a different
title where one is Miss/Mrs/Ms, or two different known genders. "Alice
Smith", "Smith", "Mrs. Smith" are one name; "Miss Roylott" and "Dr.
Roylott" are two.

A name of unknown gender learns its gender from a clear resolution: when
he/she resolves to it with the best candidate at least 0.5 ahead of the
next. The learned gender applies to every form of the name. A place
(2.3) never learns a gender. A wrong early resolution therefore fixes a
wrong gender for the whole text; the margin requirement limits this but
does not remove it.

## 8. Confidence and output

The margin of a resolution is the score difference between the first
and the second candidate. A margin below the caller's threshold means
"unclear"; the caller may then print the top candidates together
("he [Smith or Jones]") or leave the pronoun alone. Learning (section 7)
and chains (5.2) use only clear resolutions.

## 9. Evaluation

- GAP development set (2000 Wikipedia snippets, he/she only, two named
  candidates A and B): the score is F1 over A/B/neither. It measures
  names versus names and says nothing about it, they, descriptions or
  dialogue. Current: restricted 62.8, free 57.6.
- Winograd schemas (273): commonsense cases, mostly out of reach for
  rules; used to make sure nothing silly happens.
- Story gold: every third person pronoun of "The Speckled Band" (497)
  with its antecedent, drafted by a chat model and scored by
  examples/pronoun_gold.rs with alias groups for the characters. Current:
  56.9% overall; his 73, he 67, she 59, her 50, it 39. The draft is not
  yet checked by hand.

## 10. What the code does not do yet

- demonstratives this/that/these/those (1, 2.5, 4.3)
- singular they (1.1): "they" is plural only
- relative pronouns (2.4)
- clause antecedents for it (2.5)
- direct speech (2.7, 6): quotes are not detected; first and second
  person are skipped everywhere; the speaker bonus (5.1) is the only
  use of speech verbs
- binding (4.2): only reflexives are handled
- cataphora in fronted clauses (3)
- parallelism (5.7)
- pronoun chains (5.2): tried twice, once for names and once for every
  antecedent; no gain on GAP and no visible gain on the story, so left
  out until a gold file for the story can measure it
- animacy of nominals beyond person nouns (4.1): a person noun is
  excluded for "it", but "the lawn" is not excluded for "he" when
  nothing better is around, since there is no list of things yet
- hand checking of the story gold file (9)
