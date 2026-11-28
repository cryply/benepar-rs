# Where the pronoun resolver fails and what it lacks

Every miss of the current resolver on the story gold (The Adventure of
the Speckled Band, 497 pronouns, 194 misses, 61.0% correct) was read
and put into one of the groups below. The question behind the grouping
is the one that decides the rewrite: is the needed information in the
parse tree and not used, is it outside the tree, or is it in the tree
and used but weighed wrongly.

The gold is a TSV with one line per pronoun (sentence, word, wanted
antecedent, alternative) drafted by an LLM and corrected by hand; it
is not in the repository.

## Summary

| group | misses | information needed | where it is |
|---|---|---|---|
| 1 lexical semantics of nouns | 36 | person / animal / thing / abstract / time, gender | not in the tree: WordNet or a word list |
| 2 entity identity | 46 | which descriptions, names and speakers are one entity | partly in the tree (predicates, vocatives, appositions), partly lexical (woman = lady = girl) |
| 3 syntactic binding | 42 | binding domains, relative clause heads, coordinated VPs, inversion, partitives, negation | in the tree, not used |
| 4 place names | 7 | that Calcutta is a place | tree gives a cue ("to X, where ..."); a gazetteer gives the rest |
| 5 expletive and cleft it | 8 | it + copula + NP/ADJP + clause patterns | in the tree, patterns incomplete |
| 6 clause antecedents | 11 | it referring to a clause or a proposition | in the tree, not modelled |
| 7 salience and dialogue | 29 | topic of the question in an answer, local object over frequent name, generics | discourse state; the tree gives the pieces |
| 8 gold | 14 | missing aliases, time it, two wrong answers | fix the gold |
| 9 parser | 1 | "Perhaps" tagged as a noun | parser |

Only group 1, part of group 2 and group 4 need information from outside
the parse tree. Groups 3, 5, 6 and most of 7 are information the tree
already carries; the current resolver reads one flat list of NPs with a
few flags off the tree and loses the rest.

## 1 Lexical semantics of nouns (36)

The resolver knows a short list of person nouns and a suffix rule
(-er, -or, -ist, -ian, -ess). The suffix rule makes "answer", "poker",
"paper" and "saucer" people, so "he" in s199, s206, s271, s272, s274,
s283, s284 and "her" in s588 land on them (12 misses). The list lacks
"occupant" (s584, 2), has no gender for "coroner", "driver" (s46, s159,
s160: 5), and knows nothing about animals: "the reptile", "the creature"
are things, so "it" and "its" with "perch", "temper", "master" prefer
"the noose" and "my cane" (s567, s591-593: 11). "instant" is not a time
word (s535-537: 3), "inspection" is not marked abstract (s586: 1), and
"acres" is as good a possessor of "tents" as "gipsies" (s86: 2).

Needed, per noun: person or not, animal or not, male or female or
either, abstract, unit of time. WordNet lexicographer files
(noun.person, noun.animal, noun.time, noun.cognition ...) give the
first four for the whole vocabulary; gender needs a small list on top
(kinship, titles, -ess). A curated list of a few hundred words would
cover this story but not the next one.

## 2 Entity identity (46)

### 2.1 Descriptions, names and introductions (23)

The visitor is "a young lady dressed in black", then "the woman",
"the lady", "our visitor", "Miss Stoner", "Helen". The resolver has no
entity for her: each NP is a candidate on its own, and "Mrs. Hudson"
(a name, +2) beats "the woman" (a nominal) for every "her" in s28-35
(10 misses). The same with "a huge man" against "Sherlock Holmes"
(s237-238: 6), "our new visitor" (s252), "the doctor" after "sat Dr.
Grimesby Roylott" (s562), "a clever and ruthless man" (s578), "the
lady" against "a housekeeper" from a hypothetical in a quote (s221:
3).

Needed: an entity store built while reading. An indefinite NP opens
an entity; a definite NP with the same head or a hypernym head (lady,
woman, girl, visitor, client) attaches to the open entity that fits;
a predicate nominal or apposition ("I am Dr. Grimesby Roylott", "his
only son, my stepfather") merges entities; a name attaches to the
entity it is predicated of or in apposition with. Hypernymy between
person nouns is lexical (WordNet again); the rest is in the tree.

### 2.2 Vocatives and addressees (16)

"Helen!" inside the dying sister's speech is a vocative; the resolver
takes it as a mention and gives "Helen" to every "she" and "her" that
means the sister (s150, s152: 9). "Watson" in "what do you think of it
all, Watson?" is the addressee, and "Watson" has no known gender, so
it takes "her" and "she" (s225, s468: 6). "my God" is a candidate for
"It" (s149).

Needed: a vocative is an NP that is a sentence of its own ("Helen !"),
or is set off by commas at the edge of a quote, or follows an
interjection. A vocative names the addressee of the quote; it is not a
mention in the narrative. The tree marks all of this.

### 2.3 Speech attribution (7)

Inside Holmes's long explanation at the end (s578-583) every "he" and
"him" is the doctor, and "Sherlock Holmes" wins because the speaker
exclusion did not apply: the quote runs over several paragraphs and
the speaker is lost (5 misses). In s270 "he snarled" after the
doctor's own quote resolves to Holmes: the turn-taking rule assigned
the quote to the wrong party (2 misses).

Needed: quote state and speaker carried across paragraphs, and the
speaker fixed by self-introduction ("I am Dr. Grimesby Roylott, of
Stoke Moran") when the subject of the speech verb is only a pronoun or
a description.

## 3 Syntactic binding (42)

All of this is in the tree and the resolver does not look at it.

- A reflexive takes the subject of its clause, also when that subject
  is a pronoun or a relative pronoun: "he shut himself up", "which is
  itself crushed", "who threw itself", "seeing that he must adapt
  himself" (s71, s72, s82, s320, s365, s415, s498, s562: 10). The
  resolver returns nothing for a reflexive whose subject is a pronoun.
- A non-reflexive pronoun is not the subject of its own clause: "his
  breadth seemed to span it", "the morning light revealed it", "your
  very life may depend upon it", "he shook his fists at him", "there
  isn't a cat in it" (s237, s392, s449, s516, s581-584: 9).
- A pronoun in a relative clause is not the head the relative pronoun
  already binds: "the handkerchiefs which so many of them wear" (s178:
  2).
- "the top of it", "the seat of it": the pronoun is not the head of
  the NP it sits in (s395, s401: 2).
- The object of a coordinated VP in the same sentence outranks
  anything earlier: "seized the poker, and bent it", "took out my
  revolver and laid it", "reached the lawn, crossed it", "picked up the
  steel poker and ... straightened it" (s269, s273, s498, s519: 4).
- Existential and inverted clauses put the subject after the verb:
  "there broke from the silence the most horrible cry. It swelled",
  "there reared itself ... the head of a serpent" (s542, s562, s563:
  3).
- "a couple of ..." and "one of those bars": the head is the embedded
  noun, not "couple" or "one" (s360, s361, s228: 3).
- "No servant" is not an antecedent (s90: 3).
- The subject of a participial adjunct is the subject of the clause it
  hangs on: "said our new visitor, ... shaking his hunting-crop"
  (s252 is counted under 2.1).
- An appositive chain in one sentence: "his only son, my stepfather,
  seeing that he ..." (s72 "he" is counted under 4).
- A pronoun with a relative clause opens an entity: "he to whom of all
  others I have a right to look ... looks upon all that I tell him"
  (s61, s62: 3).
- Right dislocation "How long they seemed, those quarters!" (s528: 1),
  "none, save only one, who cares for me, and he" (s49: 1), possessive
  bound to the local subject "her sister had met with her fate" (s350:
  1), "with a sudden effort" between verb and object (s273 it: 1),
  "to the rope -- or so we may call it" (s483: 1).

## 4 Place names (7)

"went out to Calcutta, where ... he established a large practice":
Calcutta is seen once, with one motion preposition, so it stays a
person and takes "he" and "his" for three sentences (s72-74: 7). The
tree says "where" right after it; a relative "where" marks a place on
the first mention. "Scotland Yard" (s269) is the same problem (counted
under 2.3). A gazetteer of place names would settle both without
waiting for syntax.

## 5 Expletive and cleft it (8)

Patterns the current check does not cover: "it + copula + NP + that/to
clause" ("was it your custom to lock", "it is my strong impression
that": s126, s157), "it + copula + ADJP + for/to clause" ("would it be
possible for us": s204, "it is very kind of you": s456), "I thought it
as well that" (s306), "I take it" (s329), "it was all that I could do"
(s510), and the cleft "It was a singular sight which met our eyes"
(s553). All are tree patterns.

## 6 Clause antecedents (11)

"it" referring to a clause or a proposition: "I can read it from his
soothing answers" (s62), "what do you think of it all" (s222), "It may
stop his gossip" (s307), "it must be so" (s423), "it is for the best"
(s548), "It is a nice household" (s503), "All was as we had seen it"
(s509), "it is the worst of all" (s412), "I presume that it is
something very pressing" (s15), "all that I tell him about it" (s61),
"it may have referred to some band of people" (s177). The resolver has
only NPs as candidates, so it always answers with a noun. The gold
marks these CLAUSE; a candidate for the previous clause, taken when no
noun fits the verb, would cover most of them.

## 7 Salience and dialogue (29)

The right candidate is in the list and loses on score.

- Question and answer: the answer's pronoun means the thing the
  question asked about: "Did they find poison? The doctors examined
  her for it" (s171), "Could you swear to it?" (s155, s157), "Your
  sister asked for it?" (s360 also under 3), "Won't it ring?" (s370),
  "There have been changes? They seem ..." (s382), "It was clamped"
  (s479) (7).
- A frequent name beats the local nominal: "the will" against "a
  sheet of blue paper" (s280: 2), "the bedrooms" against "the
  sitting-rooms" (s103: 2), "the whistle" against "the next room"
  and "the lawn" (s120, s122: 3), "the scream" against "the wind"
  (s139), "the case" against "the chance" (s18), "the fire" against
  "the good sense" (s27), "the corridor" against "the other side"
  (s338), "the chamber" against "a small side door" (s351), "the
  bell-rope" against "his eye", "a dummy", "a wire" (s369, s371, s373:
  3), "the things" against "books" (s386), the rooms against "the
  dangers" (s447), the child against "our way" (s500), "the safe"
  against "the door" (s588) (19).
- Generic indefinites inside a conditional or a when-clause are not
  entities: "when a clever man turns his brains to crime" then takes
  "he" for the next four sentences (s415, s416: 4).
- Predicate nominals are excluded for he/she but not for it, so "it's
  a dummy" makes "a dummy" the antecedent of the next "it" (s371,
  counted above).

Most of this is the scoring: frequency 1.0 per earlier mention lets a
name mentioned forty times win against the subject of the previous
sentence. A rewrite should weigh grammatical role in the previous
sentence (subject, then object) and the question's focus far above
raw frequency, and give indefinites under "if", "when", "whether" no
entity at all.

## 8 Gold (14)

- Missing aliases: "this unfortunate lady" = "my sister" (s173: 2),
  "this fellow" = Roylott (s307), "the terrified girl", "the poor
  girl" = Miss Stoner (s569, s572: 4), "the schemer" = Roylott (s565).
- Time and weather "it" wanted as "the time", "that night", "the
  weather" (s8, s136, s248): the resolver returns nothing, which is
  the right answer under the spec.
- Wrong answers: s87 "their master" is the animals', not the
  stepfather's; s273 "As he spoke he picked up the steel poker" is
  Holmes, not the doctor; s543 "They say that" is generic.

## What the rewrite needs

From outside the tree:

1. A noun lexicon: person, animal, abstract, time, gender. WordNet
   lexicographer files cover the first four; gender from a list.
2. Person-noun hypernymy for entity linking (lady, woman, girl, visitor,
   client, fellow, man, doctor): the same lexicon.
3. A gazetteer of place names, or at least the syntactic cue
   "to X, where".

From the tree, not used today:

4. Binding domains: the clause of the pronoun, its subject, the head
   of the NP it sits in, the head a relative pronoun already binds.
5. Coordinated VPs, existential and inverted subjects, partitive heads,
   negative determiners, participial adjunct subjects, appositions,
   vocatives, the it-patterns of section 5.
6. Predicate nominals and appositions as identity statements between
   entities.

From discourse state:

7. An entity store (section 2.1) with gender, person, place, speaker,
   aliases, learnt while reading, instead of a flat list of NPs with
   scores.
8. Quote state and speakers across paragraphs; vocatives as
   addressees; self-introductions.
9. The focus of a question for the pronouns of its answer; the previous
   clause as a candidate for it.

With the gold fixed (14) and groups 1, 3, 4, 5 and 2.2 done, which are
mechanical once the lexicon and the tree walk exist, the ceiling on
this story is about 86%; groups 2.1, 2.3, 6 and 7 are what separates
that from 95%.
