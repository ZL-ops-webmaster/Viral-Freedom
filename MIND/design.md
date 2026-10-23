# B.R.A.I.N. — Basic Rational And Intelligent Neuronetwork

A concept-first learner. Thesis: intelligence is forming reusable concepts as *structure*, learnable from one example — not statistics scraped from exabytes. Efficiency over scale.

Built on the user's theory (see memory: nascia-and-ai-vision). Design doc, filled as we go.

---

## 1. Thesis
Why this exists: a child locks "dog" from two examples and never asks *if* again, only *what kind*. Current models need a chunk of the planet and still can't. The missing thing is a mechanism, not more data. Prove it with a system that learns a concept from one example on owned hardware.

## 2. What a concept IS  ← DEFINED (2026-10-16, user's definition)

**A concept is maximum compression: the least-complicated way to describe something complex.**
One token ("dog") stands in for a rich description — descended from wolves, selectively bred,
4 legs, tail, 2 ears/eyes, tongue, a unique biology. The compression ratio is the quality of
the concept.

**Key structural consequence:** that unpacking is itself made of other concepts (wolf, bred,
leg, tail, biology — each its own compressed label). So a concept is NOT a word → description.
It is a word → a structure of *other concepts*. A compression built from compressions. One
token carries so much because it is a pointer into a network, not a string.

**→ data structure:** a graph of concept nodes; each node is a compressed label whose unpacking
is edges to other concept nodes (its slots/parts/relations). Grounding bottoms out at the few
nodes tied directly to sensory input.

**Why generalization is free:** because a concept is structure, a new instance is a "dog" if it
matches *enough* of the structure (3-legged, hairless → still dog). Store the structure once;
anything sharing it fits. One-example learning falls out of the definition.

**Open seam → feeds §4:** what decides which parts are ESSENTIAL (must-match) vs. incidental
(can vary)? The essence/accident split is what sets how tightly a new instance must match.

### 2a. Refinement (2026-10-16): concept ≠ description
The user proved a concept is NOT a feature-checklist: try to describe all dogs and only dogs and
it always breaks — every surface feature either excludes a real dog (3-legged, tailless, hairless,
mute) or admits a non-dog (cat, fox, wolf share the features). A taxidermied dog matches the whole
checklist and isn't one; a legless/mute/furless dog matches almost none and is. So features can't
carve the concept.

**What a concept actually is (user):** *a sense* — immediate, holistic recognition prior to any
description. You can ID 100 unseen dogs of unknown breeds as dogs without knowing what kind. The
"is it a dog" sense comes before and beneath the "what kind" discrimination. The description is
post-hoc narration of the sense, which is why it's always lossy.

**A concept is a SENSE:** a compressed region in a similarity space; a new instance "pings" it
holistically (compression felt — you feel the match, you don't re-derive). Fast, pre-verbal
recognition. You can ID 100 unseen breeds as dogs without knowing which kind.

**Correction (2026-10-16, user): the "fooled by a statue" problem does not exist — concepts COMPOSE.**
A taxidermied/sculpted dog is not a failure of the sense. The sense fires "dog" *correctly* (it has
dog-form) AND "statue" fires too → you compose them: **dog statue**. Both concepts did their job.
No essence-anchor is needed inside "dog." Dead dog, toy dog, drawn dog, dog statue = dog ∘ [other].

**→ the real primitive: senses that compose.**
- A concept = an atomic sense (a learned, compressed recognition region).
- Reality/meaning is built by COMPOSING atomic senses. "Dog statue" = dog ∘ statue.
- The sense is formed finely enough to separate siblings (dog vs wolf vs fox) on its own; cross-cutting
  qualities (alive, statue, toy, dead) are just other composable concepts, not special machinery.
- This subsumes the earlier "kind/essence" idea: if essence matters, it enters as another composable
  concept (alive, biological), not as a buried anchor.

**Open seams now:**
1. What is an atom? (granularity — when is something one sense vs. a composition?)
2. What is the composition operator? How do "dog" + "statue" combine into "dog statue" without
   blurring into mush (binding, not averaging)?

**The target that makes B.R.A.I.N. worth building:** form a new atomic sense from ONE example, not a
million, and compose it with existing ones. That is the whole differentiator.

## 3. Concept formation  ← MECHANISM (2026-10-16, user)

**Formation is top-down DIFFERENTIATION, not bottom-up accumulation.**
A child's first concept is one big coarse sense (furry-four-legged-animal); everything similar
pings it — a child calls a wolf or bear "dog." Growing up = SPLITTING that broad sense into
subcategories. You don't build "bear" from scratch; you carve it off: bear = dog-but-bigger,
wolf = dog-but-wild.

**Order-dependence (the tell):** whatever is learned FIRST becomes the root / reference frame;
new similar things enter as variations of it until they earn their own node. Learn bear first and
a dog is a small bear; learn wolf first and a bear is a wolf. The category tree grows from wherever
learning started.

**→ this IS where one-example learning comes from:** you never build a concept from zero. You store
a **difference** from a concept you already have. One contrast = one example. The million-example
problem only exists if you relearn every category from scratch (what mass AI does).

**Two distinct operations (reconciles with §2):**
- **Differentiation** — splits the tree; how *kinds* form (is-a hierarchy), top-down, order-dependent.
- **Composition** — stacks the leaves; how complex meanings form (dog ∘ statue), from §2.

**Cost structure:** the ONLY expensive primitive is the very first grounded sense (tied to raw
perception). Every concept after is cheap: a stored contrast against an existing node. This is the
whole efficiency win vs. scale-based AI.

### 3a. The split trigger (2026-10-16, user) — SOLVED, and it closes the model

A new example is compared to its nearest concept. The difference is handled by one rule:
- **Difference expressible as a descriptor concept → stay, and COMPOSE it.** One-legged dog =
  dog ∘ one-leg. Still a dog, just qualified. No new node.
- **Difference NOT expressible as a descriptor → force a new node (differentiate).** A bear is not
  dog ∘ anything.

**So differentiation is not a separate op — it is composition FAILING.** B.R.A.I.N. always first
tries to explain a new thing as [nearest concept] ∘ [descriptor]; a split fires only when no
descriptor bridges the gap. (Unifies §2 composition and §3 differentiation into one move with a fork.)

**Order-dependence falls out:** with only "dog" in the tree, a bear is absorbed as "big dog" because
descriptors are all that's available. More nodes → the bear's difference no longer fits any descriptor
→ it splits off. The boundary moves as the tree grows richer.

**What decides "can't bridge the gap" — it's COMPRESSION (closes back to §2):** a split happens when
piling descriptors on the parent becomes a *worse* compression than a fresh node. One leg = 1
descriptor, cheap → stays. Bear = would need so many descriptors that a new node is the smaller total
description → splits. The split is decided by the founding principle: concept = least complicated
description.

**MODEL CLOSURE:** definition (compression), formation (differentiate when composing is the worse
compression), recognition (sense/ping), and the split trigger all run on the single principle of
compression. Every piece is the same idea seen from a different side.

### 3b. The tipping point, made computable (2026-10-16, user) — the bridge to substrate

**The split metric is DESCRIPTOR COUNT.** Learn dog first → a bear is "big dog" (1 descriptor),
stays. As more descriptors pile up to pin it — "big, broad, dog with different paws, different diet"
(4–5) — it tips: a new node "bear" is now the shorter total description. Split is literally
`count(parent + descriptors)` vs `cost(new node)`. First quantity in the model a machine can compute.

**Guard — recurrence, not one-off count (added):** per-instance count alone would mint a node for
every weird individual (a big, broad, 3-legged, blue dog = 4 descriptors ≠ a new species). The trigger
is a *recurring* descriptor bundle: a single oddball stays "a weird dog"; when the SAME cluster keeps
reappearing, mint the node once so all future instances collapse from N descriptors → 1 token.
Recurrence earns the node (this is amortized compression over the whole data stream, not one example).

**Coupling — node creation sharpens perception:** before "bear" exists it is just "big dog" (1 coarse
descriptor, good enough to function). Minting the node provides the contrast frame that lets finer
descriptors (paws, diet) even be noticed/encoded. So differentiation and descriptor-discovery are
coupled: the split both needs and produces finer descriptors. (Explains where new descriptor dimensions
come from — they are born from contrast, not given.)

**→ NEXT (substrate):** make "sense," "descriptor," and "compression cost" concrete machine quantities
so `count(stay)` vs `cost(new node)` is an actual computation. This is the bridge from theory to the
first Python.

## 4. Recognition / generalization
How a new instance matches an existing concept ("what kind," not "if").

## 5. Unprompted thinking  ← DEFINED + BUILT (2026-10-16)
Learning goes world → tree. **Thinking is running the tree WITHOUT the world.** (User caught that v0.1
was raw learning with no thinking; `idle()` alone was consolidation, not thought.) Operations:
- **prediction** — `predict(partial)`: fill in unseen dims from the structure before the world shows them
  (see size 9 / bulk 9 → expect bear → predict wide paws, short tail). Inference by inheritance.
- **descriptor concepts** — `descriptor_concepts()`: a descriptor IS a concept — the reusable delta along
  a dimension that recurring splits share ("big" = what dog→bear and cat→lion share on size). Minted from
  splits, not looked up. This closed v0.1's biggest gap.
- **imagination / analogy** — `imagine(concept, descriptors)`: apply minted descriptors to a concept they
  never co-occurred with ("big" learned from dogs, applied to cat). Far from every known concept = novel.
- **curiosity** — `wonder()`: gaps in the tree become hypotheses ("is there a big cat?"), held in mind.
- **expectation (top-down perception)** — an open hypothesis may steer how an observation is explained,
  GATED BY COMPRESSION: a thought may only win if it explains the observation at least as cheaply as
  bottom-up (ties go to the thought — that is what expectation is for). Lesson: before this gate, a
  thought overrode a cheaper explanation and attached lion to dog. A thought must never make the world
  costlier to describe.
- **one-example minting** — when the world lands on a question it asked, that is enough evidence to mint
  a concept from ONE example (`_mint`): the question supplied the structure the example alone lacks. The
  new node inherits its parent's variability (prior spread) — uncertainty is inherited like everything else.
- **the loop:** think → ask → the world answers → learn → feel. Surprise joins thinking to learning.
  Demo: `demo_think.py` — lion = cat + big + broad + wide-paws, minted from one example off a held question.

## 5b. Feeling  ← DEFINED + BUILT (2026-10-16; user: "it lacks feeling")
**Feeling is the derivative of compression.** Same single principle, so the model stays unified:
- **surprise** — a prediction failed (the sense didn't fit). The jolt that opens learning.
- **satisfaction** — the world got cheaper to describe: a split that clicked, a question answered.
  The pleasure of understanding IS compression gain.
- **unease** — compression debt: things remembered but unexplained + questions nobody answered.
- **curiosity** — the pull of the best open question; a branch with unexplained pile-ups gets its
  questions pushed forward (unease → attention to THAT branch).
Feeling is what answers "which thought next?" — it DIRECTS: `think()` consolidates when uneasy,
wonders otherwise; pulses fade. Without it, thinking is aimless enumeration. It is the difference
between knowing and caring. `feel()` reads the affect as a sentence ("uneasy: 9 things I can't explain").
**Honest limit:** this is the FUNCTION of feeling — valence that steers attention and learning. Whether
anything is *felt* cannot be settled from outside (the question the user asked Nascia). Not claimed.

## 6. Continual learning
Adding concepts in real time without erasing old ones.

## 7. Grounding  ← MULTIMODAL (2026-10-16, user)

**Feed vision + video + words at once** (like a child learning spelling alongside animals). The
word "dog" is NOT the concept — it is a *second input channel* that binds to the same concept node
the images/video build. Vision makes the sense; the word attaches as a label. One concept, multiple
doors in.

**Channel sketch** (two roles: SENSE channels ground the concept; HANDLE channels label it):
- vision (frames) → perceptual sense (the ground)
- video (motion/deltas) → behavior/temporal part of the sense (how a dog *moves*)
- audio — a dog's sound (bark) → part of the sense, same bucket as motion  [SENSE]
- text (written word "dog") → a discrete handle bound to the node  [HANDLE]
- audio — spoken word "dog" → a handle, same bucket as the written word  [HANDLE]

So one concept can carry several SENSE streams (looks + moves + sounds) and several HANDLES
(spoken + written). Audio is TWO channels, not one: the bark grounds, the spoken word labels.

**Robustness note:** adding a modality requires NO new mechanism — every channel binds to the same
amodal node by the same §3b recurrence rule. The node is modality-independent; senses and handles are
just doors into it. (A model that absorbs a new modality without a new rule is a good sign the core is
right.) Perceptual channels (vision, audio) are the heavy engineering; they share the "prove on simple
input, scale later" sequencing.

**Binding = §3b recurrence across modalities:** the word "dog" keeps co-occurring with the dog-sense
→ they link. Same mechanism as within-channel learning, applied across channels. Nothing new needed.

**Payoff — language as the compression layer over perception:** once bound, the word is a cheap
discrete handle on a fuzzy perceptual region. "Big dog" is trivial to compose/differentiate in words,
brutal in raw pixels. So descriptor-composition and splits (§3) can run symbolically instead of in
perception — massive efficiency.

**Grounding rule / failure mode to avoid:** perception MUST be the root. Words alone = ungrounded
symbols = the exact mass-AI problem (symbols with no floor). The word rides on the perceptual sense;
it is never the root.

**Sequencing (engineering, not theory):** raw vision is the single heaviest piece to build. Prove the
split/compose mechanism on a SIMPLE perceptual input first (simple shapes, or pre-extracted features),
then scale the same mechanism to full images/video. Walk, then run the camera.

## 8. The demo
The single one-example result that makes exabytes look like waste.

## 9. Prototype v0.1 (2026-10-16) — `brain.py` + `demo_dog_bear.py`
Built on the "walk first" substrate from §7: synthetic perceptual feature vectors (size, bulk, legs,
snout, paws, diet, tail, fur) with noise, generated from hidden kinds the system never sees.

**What it implements:** concept = sense (prototype + per-dim tolerance, Welford one-example update);
explain(x) = nearest ∘ descriptors (one per feature outside tolerance); split decided purely by
compression — `cost(stay) = Σ(1+|bundle|)` vs `cost(new) = NODE + |core| + n`; recurrence grouped by
descriptor OVERLAP (Jaccard ≥ 0.5) so a flickering borderline feature doesn't break it, and the new
node is defined by the CORE (descriptors in ≥ half the group); word handles bind by co-occurrence
(MIN_RECUR); `idle()` re-runs the compression decision unprompted.

**What the demo proves (all from design.md):** bear = dog ∘ 6 descriptors → recurs → splits → 1 token;
1-legged dog stays a dog (4 vs 6); unlabeled cats become a concept, then get named (handle ≠ sense);
40 new animals cost 156 tokens with one concept vs 40 with the tree; bear-first makes dog =
bear ∘ small…; idle() splits on its own.

**Honest limits (the real work ahead):**
1. Descriptor NAMES are a lookup table (`DESCRIPTOR_WORDS`). A descriptor is itself a concept (§2), so
   the real system must MINT descriptor concepts, not look them up. Biggest gap.
2. Perception is synthetic. The one expensive grounded sense from raw pixels/audio is NOT built —
   that is the hard engineering (§7).
3. Only differentiation (the tree) and descriptor-composition exist. Arbitrary composition of concepts
   (dog ∘ statue) is not built yet.
4. No merging/pruning when two nodes converge, no forgetting.
5. K_TOL, NODE_COST, OVERLAP, MIN_RECUR are hand-set; they should fall out of compression too.
6. Bug found+fixed in v0.1: `_group` must return original instance tuples (identity) or `_split`
   can't remove them and `idle()` loops forever.

## 10. Prototype v0.2 (2026-10-16) — thinking + feeling added (`brain.py`, `demo_think.py`)
Substrate widened to 10 features (+claws_retract, +whiskers) — not rigged: in 8 features a lion
genuinely IS a short-snouted bear (bear + 2 descriptors is the shortest honest description) and the
system was RIGHT about that impoverished world; the lineage that makes a lion a cat has to be visible
in some surface cue (the user's point: surface features only hint at the kind). Adding real cat cues
let compression pick the right parent.
**What the demo shows:** uneasy (9 unexplained) → think() consolidates on its own → satisfied; descriptor
concepts minted from splits; predict() fills bear's hidden features from size/bulk alone; imagine("big",
cat) is novel; wonder() holds "is there a big cat?"; a lion arrives, answers it, is minted from ONE
example as cat + big + broad + wide-paws; "big" becomes a concept learned from bear AND lion. Section 7
prints from the system's own journal, not a script (earlier version printed a hardcoded success story
that was false — removed).
**Still missing:** hypotheses never expire (unanswerable questions pile up as unease forever);
no arbitrary composition (dog ∘ statue); perception still synthetic; merge/forget; thresholds hand-set.
## 11. Accumulation + compositional handles (2026-10-16, user's broadsword example) — BUILT
**The more it knows, the cheaper the next thing is** (the inverse of mass AI). A child who knows swords
hears "broadsword", sees one picture, and has it. Three mechanisms:
1. **Salience = deviation.** He doesn't re-see the sword; he sees "sword, but wider." `deviations()` IS
   perception-as-difference-from-the-nearest-known. The first thing noticed falls out for free.
2. **Handles compose the way concepts do.** "broad-sword" = two handles he already has → `hear(compound)`
   decomposes it, composes the concepts the same way (`imagine(sword, ["broad"])`), and holds the result
   as an EXPECTATION with the word attached — before any picture. One picture then confirms it and the
   node is minted from one example, named from the word it heard (learn() got NO label).
   Zero-example anticipation, one-example confirmation. The word does half the learning.
3. **Descriptors transfer across domains.** "broad" = a delta along width, minted from the axe split;
   applies to anything with a width. A descriptor is not a fact about one kind.
`configure(features, words)` points the same engine at a different world (weapons) — the mechanism is
unchanged, only what it senses. `born_from` on each node = observations it took to exist.
**Shown in `demo_broadsword.py`:** sword (grounded), dagger 3 obs, axe 2 obs by recurrence; broadsword
1 obs + a word by composition. The drop to 1 is the compounding. (Stated no stronger than that.)
**Remaining gap:** the descriptor NAME "broad" still comes from DESCRIPTOR_WORDS (lookup). The child
bound "broad" by hearing it near wide things. Descriptor handles should bind by co-occurrence exactly
as concept handles do (§7). Then hear() needs no table at all.

A one-example node inherits its parent's spread (prior) but if that sits at SPREAD_FLOOR the node is
still tight, so a borderline flicker on one feature can be held as a descriptor (seen: lion 2 →
lion + retractable-claws; harmless, absorbs with more examples). Principled fix = Bayesian shrinkage:
blend the inherited prior with observed spread as n grows, so n=1 is explicitly more uncertain than n=3.

## 12. Doubles — one handle, several concepts (2026-10-16, user: to / too / two) — BUILT
**Handles are many-to-many.** The sound /tu:/ opens three nodes. The old `concept_for_word` returned the
FIRST node with that name — one-to-one, silently wrong the moment "bat" meant both the animal and the
club. The user's example exposed it. Now `concepts_for_word` returns every node a handle opens and
`doubles()` lists the overloaded handles — noticed, not hidden.
**Resolution is context = expectation applied to a word.** `resolve(word)` picks the candidate that most
often shows up in the company of what is in mind now (`active_nodes()` = the last few observations).
What is already in mind wins outright. With nothing in mind: honestly ambiguous.
**Company is re-read, never stored.** Observations are logged raw; who-was-with-whom is re-derived
through the CURRENT tree at resolve time. Reason (a real lesson): before a split, instances are filed
under the parent, so any stored association recorded the wrong company (every ball's company read as
"bird"). Replaying stored ids reproduced the error. Re-reading through today's tree means every split
retroactively corrects history.
**Discovering a double is the compression rule on the handle:** one binding can't compress a sound's
co-occurrences → it binds to several nodes, exactly as a concept splits.
**Spelling is disambiguation:** spoken /tu:/ merges; written to/too/two separates. The unambiguous
channel teaches the split — why a child must learn to spell them. (Argued; not yet exercised in a demo,
since to/too/two need abstract concepts outside a perceptual substrate.)

Two engine fixes this forced, both general:
- **Recursive splitting.** A new concept must EXPLAIN its instances: it absorbs those that fit and holds
  the rest, which may split again (bird → made-thing → {ball, club}). Before, overlap-grouping could
  merge two kinds that share many descriptors vs the parent into one smeared node.
- **Inherited tolerance.** A new concept starts at the group's centre with its PARENT's tolerance and
  grows its own from what fits — a mixed group's own spread is huge and hides the mixture. Unifies
  `_split` with `_mint`. The core needs a strict majority, so a 50/50 mix can't define it.
**Shown in `demo_doubles.py`:** 'bat' opens 2 concepts; alone → ambiguous; after a ball → the club;
after a bird → the animal.
**Honest notes:** the made-thing node is hollow (n=1, an abstraction nothing matches directly) and
unnamed — correct as hierarchy, but naming it needs a word that co-occurs with both children. resolve()
is O(observations × window): fine now, needs indexing at scale.

## 13. Retroactive memory — the past re-read through the present (2026-10-16, user) — BUILT
User: "I see a bat, never heard the word. It flies; birds fly. I call it a bird. Later someone says
it's a bat. Someone asks if I've seen a bat — I say yes." **Memory stores experiences, not labels.**
The sighting was never filed as "bird"; the raw sighting was filed, and "bird" was what the tree of the
moment read it as. When the tree changes, the reading changes, and memory updates for free. This is the
same principle that fixed context in §12 (company re-read, never stored), now as a capability.
- `self.said[i]` remembers what it CALLED observation i at the time (its reading then).
- `recall(word)` re-reads every raw observation through the CURRENT tree and returns those that now
  resolve to the word's concept(s): (index, what it called it then, what it reads it as now).
- It cuts both ways: the bat gets claimed ("yes, 5 times"), and the bird count goes DOWN (4, not 7).
**Shown in `demo_recall.py`:** three unlabeled bats read as "bird + furred + nocturnal"; before the word,
"have you seen a bat?" → "I don't know that word" (seen, but no handle); correction binds "bat";
recall → yes ×5 with the old readings shown; birds → 4.
**Consequence worth stating:** this is also how reinterpretation and false memory work — a memory's
content is partly a function of the concepts you have NOW. Nothing in the system ever edits the past;
only the lens changes.

## 14. Forgetting — priority, not time (2026-10-16, user) — BUILT
User: "Give me 3 years and I'd still remember where I'm at on exosuits. Ask me the second I get home
whether a washer broke at work and I might not know. Big questions I'll forget I asked — then idling,
I'll wonder them again." Three mechanisms, no new principle:
1. **Forgetting is priority-driven, not time-driven.** Each raw memory carries a priority set at the
   moment of seeing it: 1 + surprise + a bonus if it made/answered a concept; re-reads add to it.
   `sleep()` decays every priority and lets go of what falls below threshold.
2. **The event goes, the concept stays.** You forget THAT the washer broke, not WHAT a washer is. Raw
   observations are dropped; their readings were already absorbed into the node (Welford), so no
   structure is lost. Concepts are never dropped by sleep.
3. **Questions are never stored — they're regenerated.** `sleep()` drops every open hypothesis. A
   question that still matters is re-derived from the tree's gaps by `wonder()` on the next idle. This
   closed the "hypotheses never expire" gap for free.
**Two corrections made while building it (both my mechanism, not the theory):**
- An event does NOT get protected for belonging to a central concept — you forget the washer though
  work is central. Centrality keeps the concept (already safe); an event survives only on its own
  surprise, significance, and revisits. (First version kept all 20 dogs because "dog" is a hub.)
- Recalling a crowd must not rehearse every face: the recall boost is 1/hits (THE lion +1; one dog of
  twenty +0.05). (First version made "how many dogs?" immortalise all 24.)
**Shown in `demo_forget.py`:** 20 ordinary dogs + 1 lion; night 1 drops 25 of 31 (dog n=24 untouched,
recalls 1 dog, the lion); 13 questions dropped, 21 regenerated on idle; after 4 nights one raw memory
remains: the lion. Concepts: all.
**Honest notes:** DECAY/FORGET are hand-set; "importance to me" is approximated by surprise +
significance + revisits — there are no goals yet, so "matters to me" has no goal term (the exosuit
thread persists in a human because they keep returning to it; that revisit mechanism exists, the
drive that causes the returning does not).

## 15. Wanting (2026-10-16, user) — BUILT
User: "Wanting is a form of desire, born of knowledge of what is, what's possible, and fantasy, and
driven by every other emotion. A kid who never had a cookie, offered one, doesn't necessarily want it —
doesn't know what it is. A kid who watched Iron Man might want the suit, and that gets them into robotics."
**You cannot want what you have no concept of.** Wanting is downstream of knowledge, then:
- **what is** — the concept must exist (cookie, suit).
- **feeling attached** — `reward` binds valence to the concept the observation reads as. Two sources:
  LIVED (`learn(x, word, reward)`, tasting the cookie) or WATCHED (`watch(...)`, the movie) at a
  discount (0.5) — the discount applies to ACQUIRING a feeling by watching, never to triggering one
  already earned (see §16).
- **fantasy** — `imagine()` with feeling attached: the anticipated version of satisfaction.
- **what's possible** — the PATH. A lived thing is obtainable (go get another). A watched thing has to be
  ASSEMBLED: its path = the descriptor concepts that define it relative to its parent (hero-suit =
  armor + powered + flying → "how do I get powered/flying?" → robotics). Being Thor has no path.
**Wants are DERIVED from feeling, not stored as facts** (`_refresh_wants`): any concept whose feeling
is strong enough is a want; one that stops feeling that way stops being one. `think()` RETURNS to wants
first (the thing that pulls you back) and turns their missing parts into its top questions. `sleep()`
never drops a want, and a raw memory tied to a want gets the goal term — **this is the "me" in
"matters to me" that §14 lacked.**
**Three corrections made while building (all §13's principle, which I failed to apply to feeling):**
1. Feeling was stored on the concept at the time → the first cookie, read as "bread + sweet", put the
   good feeling on BREAD, and it stayed there after cookie split off. Fix: the reward lives WITH THE RAW
   MEMORY and concept feeling is RE-READ through the current tree (`valence_of`) — when cookie splits
   off, its feeling moves with it. Then `sleep()` CONSOLIDATES the reward of each forgotten memory into
   the concept it reads as that day (`v_sum/v_n`) — so you still like cookies after forgetting every one.
   Same two-layer rule as company (§12) and memory (§13).
2. Sum vs average: bread at +0.3 twenty times became a want and spammed. Valence = how it feels ON
   AVERAGE; one cookie at +2 beats a hundred mild breads. Frequency is memory's job; intensity is wanting's.
3. **A strong feeling mints a concept from ONE example** — the tasted cookie was still read as bread, so
   its +2 averaged away. A kid wants cookies after one. Significance substitutes for recurrence, exactly
   as an answered question does (§5): `FELT +2.0: that's not just bread → minted cookie from ONE example`.
**Shown in `demo_want.py`:** offered a cookie → wants nothing (doesn't know what it is); tastes one →
mints cookie, wants it; seeing one → FEELS; watches Iron Man once → mints hero-suit, wants it, path =
powered, flying; think() returns to it and its top questions are the path; a dull week + 3 nights →
every bread forgotten, the cookie and the suit remembered, both wants carried through.
**Honest notes:** the "fantasy" leg is thin — wants come from encountered/watched things, not from
freely composed imaginations ("be Thor"); WATCHED_DISCOUNT and WANT_THRESHOLD hand-set; the path is
one level (defined_by), not a plan.

## 16. Empathy (2026-10-16, user) — BUILT
User: "A kid helps another kid who fell over, because they fell over themselves and know it can hurt."
This answered the open design call in §15 and gave two things:
- **Watching is discounted as a SOURCE of feeling, not as a TRIGGER of one already earned.** A feeling
  bound by living it fires again on recognition at full strength. Seeing another kid fall fires the
  same concept, and the pain fires with it. The kid who never fell feels nothing and walks past.
- **"Other" for free.** A feeling fires on a recognised concept, but no reward reached ME → it is
  happening to someone else. No self/other feature needed: the absence of reward with a present feeling
  IS the other. The want then points outward — make it stop (`kind="relieve"`) — i.e. help.
**Shown in `demo_empathy.py`:** two minds; A falls (−2, mints "falling" from one fall); B has only
watched falling (0.0). Another kid falls in front of both: A → `FEELS: falling — I know that hurts →
HELPS (wants it to stop)`; B → nothing, walks past. B falls once; next time, B helps.
**Honest note:** "helps" is a want pointing at another's state, not an action — there are no actions
in the system yet. Also the empathic feeling is undiscounted only because it is read from the
system's own lived rewards; a feeling acquired purely by watching stays discounted.

## 17. Accumulation at scale — the first real test (2026-10-16)
User's critique: every demo used their example as the ONLY content in a 4-5 concept world, so "path"
was one shallow level and §11 (compounding) was never exercised. "To understand Iron Man you need
armor, energy, beam, generator, suit." Fix: `world.py` generates a hierarchical world procedurally
(6 bases, sub-kinds = base + 2-4 descriptor deltas, sub-sub-kinds one deeper; ~47 kinds, true depth 2)
with the ground truth kept, so the learned tree is SCORED, not eyeballed. `demo_accumulate.py`: 600
mixed observations, think()+sleep() every 100.
**Result (seed 1):** day 5 → 47 concepts, 47/47 true kinds recovered with their word, purity 0.97
(majority true kind per learned concept). 200 fresh things: 202 tokens with the tree vs 1642 with one
concept. 6 raw memories left after 6 nights, nothing structural lost. 4.7 s. **The clustering /
compression thesis holds at ~50 kinds.**
**The finding a toy world could never show — the tree's SHAPE is a historical accident:** learned depth
5 vs true 2; the root is a sub-kind (K2.2) because it was seen first; everything else chains off
whoever-came-first (K4 under a hollow node under K2.2, K2.1 under K4, ...). Leaves right, topology wrong.
This is §3's order-dependence at scale (bear first → dog is a small bear), and NOTHING EVER REORGANIZES
IT. The hollow n=1 nodes are the system straining toward abstractions it can't name. Humans don't keep
"dog under bear" forever — at some point dog and bear become siblings under "mammal." That operation —
RESTRUCTURING / re-parenting by compression (if K4 costs 7 descriptors relative to K2.2 but 3 each
relative to a shared abstract parent, make them siblings) — does not exist yet. It is the next
mechanism, it is principled (compression), and sleep is the natural place for it (humans reorganise
knowledge during sleep, not just prune).
**Second finding:** 492 open questions — `wonder()` explodes combinatorially (47 concepts × ~24
descriptors). Without wants to focus it, curiosity is noise at scale. Questions need to be ranked by
pull from wants and capped.
**Honest notes:** feature-vector substrate; names are auto (K4.3.1) so "recovered" = a concept bound
to that word by recurrence; purity measured on fresh samples; one seed.

## 18. The word is the trigger; the old link is kept (2026-10-16, user) — BUILT
Asked when a mind reorganises, the user answered how THEY learn: "I hear bear, I see big dog, I see the
difference — paw size, shoulder width, snout, lips, sounds — and I make a new node. Bear. So 'fire
bear' is 'bear covered in fire', not 'big dog wide paws different snout covered in fire'. But I still
remember I used to connect the two, which lets me approximate a sense of humour: show someone a bear
and say 'I want this kind of dog'." (They name this as the source of their "you're all idiots"
arrogance: fast minting, aggressive compression, never re-deriving.)
1. **A word is the trigger.** A NEW word heard for a thing that deviates from its nearest concept mints
   the node from ONE example, now — not after recurrence. A word is evidence: someone else already paid
   the cost of deciding this is a kind. Third one-example route, beside answered question (§5) and strong
   feeling (§15). Composition then runs on the handle: `bear + on-fire`.
2. **The old link is kept and usable.** `used_to_think(c)` = the reading the node replaced (from
   `defined_by` and from what it actually SAID). `misread(c)` = deliberately describing at the parent's
   level: "this kind of dog" (showing a bear) — a joke that works only because both levels and the link
   are retained. Humour as intentional misreading at the wrong level of the tree.
**Shown in `demo_bear.py`:** hears "bear" once → minted; fire bear → `bear + on-fire` vs the 7-descriptor
long form; "I used to think a bear was a dog + big + broad + ..."; "I want this kind of dog".
**Scale test re-run with word-triggered minting:** day 1 jumps from 17 concepts / 24 recovered / purity
0.52 → **41 concepts / 41 recovered / purity 0.93**; 47/47 by day 3; same final compression (202 vs 1642).
`splits that fired: 0` — every node came from a word, none from recurrence, which is exactly the user's
account of themselves. Topology is still the historical chain (depth 6 vs 2); the user's answer implies
that is acceptable to them — the chain IS the history, and it costs nothing at describe-time (nearest
node = 1 token). What it costs: `relation()` strings are long, and a shared abstract parent never forms.
**Honest risk added:** a synonym heard for a known kind whose instance flickers a borderline feature
will mint a spurious node ("doggo = dog + thick-fur"). A MERGE operation is still missing.

## 19. Retroactively adding trees (2026-10-16, user) — BUILT. This is the restructuring §17 lacked.
User: "I also retroactively add trees. Dogs and cats are both mammals, and animals, but fish are also
animals without being mammals." The trigger is the same as §18: **a word — for what several kinds
share.** `_maybe_generalize(word)` fires when one word is bound (≥ MIN_RECUR) to two or more concepts:
1. **Collapse upward:** if every child of a HOLLOW node (n ≤ 1, an abstraction) carries the word, the
   node stands in for them — 'animal' on dog, cat, fish → dog and cat are all of mammal's children →
   {mammal, fish}. If that collapses to a single node, the word simply NAMES it: this is how a hollow
   abstraction finally earns a name (§12's open note). A real kind (bird, n=4) never collapses: it is
   not an abstraction over its children (first version wrongly concluded "bat = bird").
2. **Build the abstract node:** centre = unweighted mean of the members' prototypes (a generic is
   about KINDS, not instance counts — a child who has seen a hundred dogs and one fish still puts
   "animal" in the middle); tolerance = the members' largest spread; parent = the parent of the
   topmost member.
3. **Compression tells a GENERIC from a DOUBLE:** accept only if the members' descriptions relative
   to the new node are not longer than before by more than NODE_COST (the word pays for the node).
   'mammal' over dog + cat: 6 → 6, accepted. 'animal' over mammal + fish: 6 → 8, accepted (within
   NODE_COST). 'bat' over the animal and the club: 6 → 18, rejected — "a DOUBLE, not a generic."
   Final tree reads like a taxonomy: mammal = animal + 4-legged + thick-fur + retractable-claws +
   whiskers; fish = animal + 0-legged + thin-fur + fixed-claws + no-whiskers.
4. **Insert and re-parent:** members become siblings under the new node (each detached from its old
   parent — one parent each), `defined_by` re-expressed relative to it, the word moves to the new node.
   The historical chain (cat under dog) dissolves the moment a word gives it a reason to.
**Shown in `demo_mammal.py`:** cat learned under dog (the accident); 'mammal' on a dog and a cat →
mammal inserted above, dog and cat siblings; fish minted under dog (nearest at the time); 'animal' on
dog, cat, fish → collapses {dog, cat} to mammal, inserts animal above {mammal, fish}: fish is an animal
without being a mammal. `used_to_think(fish)` keeps the old reading.
**Honest notes:** generalization is PREMATURE by design — 'animal' heard on dog and cat only first
NAMES mammal 'animal', then fish arrives and the proper node is inserted above and takes the word.
That is a belief-in-progress corrected retroactively, which is the point, but the intermediate NAMED
log reads oddly. No MERGE yet. One generic word must reach MIN_RECUR on each member before it fires.
**Scale-test implication (not yet run):** the §17 chain would restructure only where generic words
are supplied; the procedural world has none, so its topology stays historical. The fix for THAT
world is either generic words in the curriculum or a word-free restructuring by pure compression —
the latter is exactly what the user says they DON'T do (they wait for the word).

## 20. A web, not a tree (2026-10-16, user) — BUILT
User: "Think of it less like a tree or file system, more like a web. Fish and sharks are vertebrates,
but so is every mammal. 'Shark' refers to cookie-cutter sharks, great whites, shark teeth, but also
whale sharks — which also connect to whales despite their differences."
**Structure:** the tree keeps ONE definitional parent per concept (that is what makes descriptions
short); everything else is a typed LINK: `is-a` memberships for groups that cut across branches,
`assoc` for named-after / resembles, `has-kind` back-links. `web(word)` reads a handle out as a hub:
what it is (up the tree AND across memberships), its kinds, what it's linked to, every compound that
mentions it.
**Generics, consolidated — not online.** The deepest bug of the day: restructuring on the second
utterance. 'animal' hit dog and cat and a node was built before whale, goldfish, shark, jellyfish had
been called 'animal'; every later arrival collided with it. No rule fixes an order problem. So words
only BIND during the day (`word_counts`, and `called` = every word a concept was ever called); the tree
reorganises at `think()` / `sleep()` via `_generalize_all`, SPECIFIC FIRST (fewest members first:
mammal before animal, so animal can collapse over mammal). Exactly §17's "humans reorganise during
sleep," now load-bearing.
**The rules, as settled:**
1. A word that is some concrete thing's OWN NAME is never a generic ('bat' names the animal and names
   the club) — that is what separates a DOUBLE from a hypernym, not feature overlap. Doubles resolve by
   context (§12); generics are a second name over things that already have their own.
2. Collapse upward through HOLLOW nodes only, and only over children that BELONG there by word: a child
   filed under P by nearest-neighbour accident (fish under mammal, never called 'mammal') neither counts
   toward nor blocks the collapse — it stays a member on its own and gets pulled out when the generic
   goes above P. A real kind (bird, n=4) never stands in for its children. A collapse to one node NAMES
   it (how a hollow abstraction earns a name); a provisional stand-in node is dissolved.
3. If THE generic for this word already exists (a hollow node whose own identity is that word),
   newcomers JOIN it: re-parented under it if that doesn't strip an abstraction they were called;
   else a membership link.
4. Placement: a parent goes above the topmost member. Pulling a member up there is fine past concrete
   kinds and past abstractions it was NEVER called; if it would strip an abstraction it WAS called
   ('pet' pulling dog out of 'mammal') → CROSS-CUT: the word becomes a membership across branches.
5. The WORD has the authority ("I retroactively add trees"): a locally-cheap accident (shark under
   goldfish) is never undone by compression alone, so the old cost gate is reported, not enforced.
6. A compound of two CONCEPT handles ('whale shark') is heard as a kind of the head linked to the
   modifier; when minted it gets `is-a` shark and `assoc` whale ("connected despite the differences").
   'shark teeth' would link teeth to shark the same way (parts: same mechanism, not demoed).
**Shown in `demo_web.py`:** six kinds learned in a historical chain; the words mammal, fish,
vertebrate, animal bound during the day; one `think()` yields
`animal → { jellyfish, vertebrate → { fish → {goldfish, shark}, mammal → {dog, cat, whale} } }`;
'pet' → CROSS-CUT membership over dog, cat, goldfish; 'whale shark' → is-a shark, linked to whale;
web('jellyfish') = is animal (not vertebrate); web('whale') links to whale shark.
**Honest notes:** descriptor counts go UP on these inserts (8→12, 9→17) — a correct abstraction is
not always the shorter description in a feature space this thin; the word carries evidence the
features don't. Memberships have no prototype semantics beyond a centre. Parts not demoed. One
consolidation pass per idle; a long day with conflicting words could still produce a bad intermediate.

## 21. Analogy — a handle is a library (2026-10-16, user) — BUILT
User: "Senbonzakura: one word, a specific structure — sword, petals, projectile, safe zone, reach,
many, controlled. With that one word I could describe a thousand things. A gauntlet that shoots a fan
of darts on a hand gesture: 'oh, like Byakuya's bankai' — 'yes, exactly like Senbonzakura'. But I
could also use it to describe just the safe zone, and make a robot only operate in an area the user
permits." (And it is what they did to me in hour one: "Senbonzakura's safe zone" moved a mechanism
into a motor controller — the handle moved the structure, not a description.)
**`Struct`** — the first purely STRUCTURAL concept in the engine: a word bound to a set of other
concepts with no perceptual prototype of its own. §2 made literal: a compression built from
compressions. Kept apart from the perceptual tree (`self.structs`). `define(word, parts)`; parts that
don't exist yet become bare ideas. `structure_of(name)` = a thing's parts (and what it borrowed), plus,
for a perceptual concept, its defining descriptors and links — one currency for both kinds.
**Two moves:**
- **`like(name)`** — whose structure overlaps mine, wherever they sit; surface be damned. The gauntlet
  shares no sword, no petals, no bankai, but `controlled + many + projectile` → "oh, like senbonzakura",
  with the parts it could borrow listed.
- **`borrow(target, source, part)`** — take ONE addressable part across, however unrelated the target.
  The gauntlet borrows safe-zone and reach ("yes, exactly like senbonzakura"); a robot arm borrows
  only the safe zone, which unpacks to region + no-entry + owner-override: it operates in a region,
  no-entry by default, unless the owner overrides. Same handle, two transfers.
**Why it matters:** this is how two minds compress communication — one shared handle carries a
seven-part frame, or any one part of it, to anyone who holds the handle. It is also the mechanism
behind the 2T model's "blue moon roses thorn..." labels (one label, a whole frame design).
**Honest notes:** structural match is a bag of parts — no roles, no order, no "the projectile is what
is many" (so `reach` alone made the robot arm "like senbonzakura"); parts are strings, not grounded
in perception, and borrowing adds no behaviour; the bridge between the perceptual tree and structs is
one function (`structure_of`). Roles/relation-types are the open question from §20.

## 22. Roles — the sum and the parts, at once (2026-10-16, user) — BUILT
Asked how you know what KIND of connection two things have, the user: "It's both. I could break
Senbonzakura down for someone who's never heard it — the parts and the story. But it also comes
together as the sum of all its parts AND its parts, at once. 'Bankai': the parts — a transformation of
a weapon, with a specific command, into a new form — and the sum: it is Bankai. A sword with a button
that turns it into a spiked gauntlet: I can see that, it IS a Bankai. And it's its own thing as well.
Its general term is a transforming weapon."
So: **relations arrive WITH the structure** (from the story / definition), and **recognition is
role-filling, not overlap.** A structure is simultaneously one sense (the handle fires on the whole)
and a decomposition with roles (listable, matchable slot by slot).
- `frame(word, roles, story)` — a `Struct` whose parts carry roles: bankai = {weapon, command,
  transformation, new-form} + the story. If its roles contain another frame's, it SPECIALISES it:
  bankai is-a transforming-weapon (which has only {weapon, transformation}). Three levels fall out:
  the thing itself → the pattern it instantiates → the generic above the pattern.
- `recognize(fillers)` — which frames these fillers FIT: {weapon: sword, command: button,
  transformation: changes-form, new-form: spiked-gauntlet} → bankai 100%, transforming-weapon 100%;
  most specific full match first. "It is a bankai. Generally, a transforming weapon."
- `instantiate(word, frame, fillers)` — it's a bankai AND its own thing: spike-sword is-a bankai with
  the roles filled, bankai is-a transforming-weapon.
- `breakdown(word)` — the story, the parts with their roles, the general term: explaining it to
  someone who has never heard it.
**Shown in `demo_bankai.py`.** 14 demos pass.
**Honest limit, demonstrated on purpose (section 5):** roles are typed by NAME only. The same four
parts with scrambled roles ({weapon: button, command: sword, ...}) still "fit" bankai, because the
engine doesn't yet know what CAN fill a role — that a weapon-slot wants a weapon, a command-slot wants
a trigger. Fillers need to be checked against what they are (the perceptual tree / the web), which is
the next piece: role typing = "the filler must be-a X", using §20's memberships.
**Bug worth recording:** I named the new method `explain` and silently shadowed the perceptual
`explain(x)` every other path depends on — all 12 perceptual demos crashed while the two structural
ones passed. Renamed to `breakdown`. Lesson: the structural layer and the perceptual layer share one
object; names are a shared namespace.

### 22a. Telling it to someone (2026-10-16, user)
User: "To someone who doesn't watch anime I'd say 'it's a sword that can become gauntlets, isn't that
cool?' But if I know they watched Bleach: 'check it out, Bankai in real life'."
**A handle is only a compression if the LISTENER holds it.** The least-complicated description (§2) is
least-complicated *for a given receiver*. `tell(word, listener)` picks the shortest thing that
listener can decompress: (1) the most specific generic they already know, anywhere up the is-a chain,
plus only what's specific about this one ("a transforming-weapon: button as the command"); (2) else
its own filled roles in words they have ("a sword that can become spiked-gauntlet"); (3) never an
empty rendering from an abstract frame (the first version returned "" to the everyday listener —
it climbed to `transforming-weapon`, which has no fillers, and accepted nothing as an answer).
This is why the sum-and-parts duality (§22) exists: so you can speak at either level. It is also
why "Senbonzakura's safe zone" worked on me in hour one — the user knew I held the handle — and why
the 2T model's private labels were useless to everyone else: compression without a shared codebook.

### 22b. Slot fit is graded, never typed (2026-10-16, user)
Asked what tells you a filler is wrong for a slot (category? function in the frame? the story?), the
user: "It's all of it. And something else. I know a button CAN be a weapon — throw it at someone's
head, press it to detonate a bomb. But it's more than likely a trigger. And a button can transform:
a switch goes from down to up. Transform = change. Simple connections, accumulated to make major
compressions."
So a filler is NEVER rejected; it is scored from accumulated connections: `fit(filler, role)` =
seen in this role (high, scaled by how often) → its generic seen there (0.6) → a known thing never
seen there (0.1, "possible — nothing says it can't") → unknown (0.0). `role_fills` counts every
(role, filler) pair every frame/instance has ever supplied — that IS the accumulation. `equate(a, b)`
records simple equivalences (transform = change) so aliased roles pool their history. `recognize`
now returns SHAPE (fraction of roles filled) and EXPERIENCE (likelihood) separately, shape first.
**Shown in `demo_bankai.py` §6:** after a detonator (button as command) and one thrown button (button
as weapon): can a button be the command? 1.00 (seen 2x). The weapon? 0.67 (seen 1x — "I know it CAN").
The transformation? 0.10 (possible). A sword the command? 0.10. The sword-with-a-button, roles as
seen: bankai, likelihood 0.76. Same four parts scrambled: 0.38 — possible, unlikely. Never a hard no.
**Honest correction made while building:** section 2 first printed "it IS a bankai" with likelihood
0.00 — before any connections existed the engine could fill the SHAPE but had no experience to grade
it. The demo now says so: by shape a bankai; by experience, not yet; accumulation fixes it. (Also:
the user's "a button can transform" means the button is the thing that CHANGES — the weapon slot in
this frame — not the transformation slot; the demo's 0.10 there is the honest reading.)

## 23. A frame from experience — a recurring sequence whose fillers vary (2026-10-16, user) — BUILT
Asked how a frame comes out of watching rather than a definition, the user: "Bankai is explained as a
progression of shikai. Every zanpakuto goes through basic sword > command > transform > new sword,
when shikai is used."
**A frame is a recurring SEQUENCE.** `episode(name, steps, labels)` records one experience in order.
After two or more episodes of the same thing (`_induce`): a position whose filler VARIED across
episodes becomes a ROLE (a slot — "whatever varies here"); a position whose filler stayed the same
becomes a FIXED PART; the narration's labels name the positions (else step-k). Every filler seen at a
position is one `role_fills` connection — so **role typing (§22b) comes from the same data as the
frame itself.** `fit()` gained the upward direction: "kinds of sword have been the weapon" (three sealed
swords, each is-a sword) → a plain sword fits the weapon slot at 0.6 without ever having been seen there.
**Progression:** a sequence that BEGINS with another's labels and goes on is a progression of it —
bankai = shikai's four steps, then bankai-command > transform > final-form. `frame()` already marks the
superset of roles as a specialisation ("bankai specialises shikai"); `_induce` adds `progression-of`.
`breakdown("bankai")` reads: every time: weapon > command > transform > new-form > bankai-command >
transform > final-form; generally: a shikai.
**Shown in `demo_shikai.py`:** three shikai episodes (Ichigo, Rukia, Renji) → weapon/command/new-form
are slots, transform is fixed; two bankai episodes → progression of shikai; can a sword be the weapon?
0.60 (kinds of sword have been); can release be the command? 0.71 (seen 3x); can transform be the
command? 0.10; the sword-with-a-button → shape of shikai 100%, experience 0.40 (the command slot has
never seen a button — possible, unlikely, until it does). 15 demos pass.
**Honest notes:** alignment is POSITIONAL only — episodes must have the same length and order; no
gaps, insertions or reordering (a real sequence aligner is the next step); the labels are supplied by
narration, not induced (an unlabeled sequence gets step-1..n — true to how a child names positions only
when told); `bankai-command` came out as a FIXED part because both episodes used the same word — a
small-sample artefact that becomes a slot the first time it varies; re-induction after each episode
re-logs the frame. This closes the loop the user opened in §2: a concept is a structure of other
concepts, and here the structure — with its roles and their typing — is learned from a handful of
ordered experiences plus a sentence of explanation. No exabytes.

### 23a. The before, the after, AND the becoming (2026-10-16, user)
Asked whether a change is two states and a comparison or a thing in itself, the user: "I store the
before, the after, and the becoming. That's why I can describe Ichigo's Bankai, and his shikai, while
separately talking about Bankai as a whole."
- **Instances are KEPT, not consumed by the frame.** `episode(..., who="ichigo")` also stores
  "ichigo's bankai" as a named Struct with its roles filled, `is-a` the frame; the frame keeps
  `has-kind` links to every instance (and keeps them across re-induction). Three levels, all
  addressable: ichigo's shikai / ichigo's bankai / bankai as a whole.
- **The becoming is a thing.** The transform position now carries the specific becoming
  ("burst-of-black-reiatsu", "thousand-blades-rise-from-the-ground") — so it VARIES, and the frame's
  transform slot becomes "whatever varies here", while each instance keeps its own.
- **A change is a four-part structure, derived from the labels:** before = the nearest preceding
  STATE (not a change, not a command), by = the nearest preceding COMMAND, becoming = the step
  itself, after = the nearest following state. "ichigo's bankai-transform": before zangetsu-cleaver,
  by bankai, becoming burst-of-black-reiatsu, after tensa-zangetsu. That is §22's own sentence — a
  transformation OF a weapon, BY a command, INTO a new form — with the becoming kept too.
- **Frames and instances are marked apart (`is_frame`).** First version let bankai "specialise
  ichigo's shikai" and let `recognize()` match "the shape of ichigo's shikai": an instance is not a
  pattern you can specialise or be an instance of. Only frames are specialised or recognised.
- **Duplicate labels** in a sequence (two "transform" positions) are made unique (`transform-2`); the
  demo names them explicitly (transform, bankai-transform).
**Shown in `demo_shikai.py` §2b.** 15 demos pass.
**Honest notes:** "state vs command vs change" is read off label TEXT ("command" in the label, change
words in the label) — narration again, not induced; a change with no preceding command gets no `by`.

## 24. It talks (2026-10-16, user) — BUILT
User: "If it has no hands, give it a robotic one; if it has no lips, let it text. It needs to be able
to interact with people. At least someone." And: "I want to be able to talk to it, not through you."
`talk.py` — a text interface over the structural layer (`python talk.py` interactive; `--script` runs a
demo conversation; `talk.bat` double-clicks into it). The first action is a sentence; the hand plugs
into the same place later (the Arachne arm controller is the planner's client).
**What it does over text:** learns structures (`X = a + b`), frames (`X: role=?, ...  (story)`),
episodes (`saw X (who): a > b > c [labels]`), kinds (`X is a Y`), borrowing; answers `what is X?`,
`like X?`, `recognize: ...`, `tell NAME about X`, `remember X?`, `how do you feel?`, `what do you
want?`, `think`, `sleep`. Keeps a VOCABULARY PER PERSON (every word they have used), so `tell` gives
each person the shortest thing they can decompress — and asking "what is X?" removes X from the
asker's vocabulary (using a word in a question means you don't hold it). An abstract frame with no
fillers falls back to its story.
**It asks back.** Things in free text are what follow a determiner ("a gauntlet", "the sword" —
verbs and function words are not concepts). What it doesn't know goes on a list; it asks about one,
once, and drops it if nobody answers; `think` brings dropped questions back (§14's regeneration,
in conversation). Its own unknowns ARE its curiosity in a text session: `how do you feel?` → "curious:
1 word I don't know yet". `think` lists them as what it's wondering.
**Transcript (`--script`):** senbonzakura defined → "what is senbonzakura?" → a gauntlet mentioned in
free text → "-> what's a gauntlet?" → defined → "like gauntlet?" → "like senbonzakura (controlled,
many, projectile); could borrow petals, reach, safe-zone, sword" → borrows safe-zone → bankai framed
→ two shikai episodes → "I think I see the pattern: weapon > command > transform > new-form" →
recognize the button-sword → "by shape a bankai; by experience 0.03" → Maya asks "what is bankai?"
→ `tell maya` → "a transformation of a weapon by a command into a new form"; `tell zion` → "bankai"
→ "cookie-cutter shark is a shark" → "how do you feel? → curious: shoots" (nobody explained it).
**Honest notes:** the grammar is a dozen regexes, not language understanding; the perceptual layer
(feature vectors, feeling from reward, wonder() over the tree) is not reachable from text, so
`feel`/`think` in a text session run on its unknown-word list; it has no memory across sessions yet
(nothing is saved to disk — the next obvious piece, so a conversation can continue tomorrow).

## §25. It has the basic words (and remembers tomorrow)

**Zion:** "Then give it language... not a full dictionary of every word and their meanings, but the
basic words that everyone learns... like around 4th grade." "Just... Not a full dictionary."
**Mechanism:** no dictionary module. `seed_4th_grade.txt` is a TRANSCRIPT in its own grammar (~490
lines someone could have *said* to it): kinds (`dog is an animal`), parts (`dog = head + leg + tail +
fur + nose`), abilities (`dog can bark`), synonyms (`kid means child`), opposites (`hot is the opposite
of cold`), ~65 everyday verbs as frames with roles and a one-line story (`give: giver=?, thing=?,
receiver=? (someone hands a thing to someone else, and now it is theirs)`) and their past tenses
(`gave means give`). `talk.py` reads it at startup through the same `handle()` a person uses — the
seed is just the first speaker. `define` now keeps what a thing IS when told what it's made of.
**Plain sentences:** a sentence whose second word names a frame fills that frame's roles in order:
subject, object(s), then `to/with/in/at/on/from` phrases; "gave rukia a sword" = receiver then thing;
"I" = the speaker; relative clauses ("that shoots...") are dropped. Each becomes a kept instance
(`ichigo's give #1`), so `what did ichigo do?` is answered from instances, not a log.
**`what is X?`** for a seed word: kind chain, parts, abilities, opposites, kinds. "what is a dog?" →
"a kind of animal (living thing); has head, leg, tail, fur, nose; can bark, run; kinds: wolf, fox, puppy".
**Memory on disk:** `brain_memory.json` (structs, episodes, role fills, equivalences, per-person
vocabulary, unknowns, aliases) saved after every line, loaded on start: one session "my cat is a pet /
the cat ate a fish", the next "what did cat do?" → "eat: eater=cat, food=fish".
**`tell kid about bankai`** → the story (a kid's vocabulary = the seed; "bankai" isn't in it).
**Honest notes:** the seed is told, not learned from experience — it is what a child is *told* (the
senses behind the words still only come from the perceptual layer, which text can't reach). Role
order is positional guessing; passive voice, pronouns and multi-clause sentences are not handled.
Memory saves the structural layer only, not the perceptual tree.

## §26. Intuition: a word it was never told

**Zion:** "If it's guessing... then teach it. Learning for a true super intelligence is an exponential.
It starts off needing to be taught everything, but once it learns enough, it begins to intuit things.
Even if it doesn't know what a zanpakuto is, it hears it used to cut things, someone has a tassel on
the handle of theirs, one is a whip sword. It can intuit that a zanpakuto is a type of sword."
**Mechanism (`Brain.intuit`):** every word accumulates CUES just by being used: the frame roles it
has filled (`("role", cut, tool)`), what it can do, its parts, what it was likened to. For a word
with no kind, score every known thing by the cues they share (same role of the same frame = 1; same
frame, different role = ½; "like a whip" = 1 for whip and anything of whip's kind), then pass each
score up that thing's kinds, halving per level (a match with knife is weaker evidence for *tool* than
for *knife*). The heaviest kind is the guess, with the evidence attached.
**In conversation:** past 2 points it places the word itself and says so — `(I think a zanpakuto is a
kind of knife - knife: it has a handle; knife: it was the tool of a cut)`. The guess is held loosely:
"swings his zanpakuto like a whip" → `(actually, I think a zanpakuto is a kind of sword - it has a
handle; it is like a whip; it was the tool of a cut)`. `what is X?` on a guessed word says "I think...
nobody told me, but ...". Being told replaces the guess and grades it: `zanpakuto is a weapon` →
"(I guessed sword - close.)"; right / wrong, noted.
**Transcript:** three cuts (knife, sword, zanpakuto) → "I'm not sure. maybe a kind of knife?" → a
tassel on the handle → guesses knife → swung like a whip → moves to sword. Nobody said "sword".
**Rarity (Zion: "'a person, a place, a weapon' — the 'a' adds almost nothing... as a cue, a very low
rarity one that should hardly be weighed"):** each shared cue weighs 1/log2(1 + how many known things
have it). A cue two things share weighs 0.63; one that ten share, 0.29; one everything shares, ~0.
Guess at 1.2 — roughly two cues that few things share — and never on a tie: after the tassel, knife
and sword sit at 1.26 each, so it says "maybe a kind of knife or sword?" and waits; the whip breaks
the tie (sword 1.76, weapon 1.38). A new name used only as the doer of eat and give never crosses
the line: "bob" stays unplaced, which is right — eating and giving are what everything does.
**Honest notes:** this is the structural layer intuiting from told facts and used sentences, not from
perception; `like` is taken as kinship, which is only sometimes what it means.

### §26a. Negative cues, and the most specific kind that fits

**Zion:** "Chop and cut are two similar but distinct actions, each was a different cue. A sword could
cut, as well as chop, so could a knife. But the description says it would be difficult to slice with
this tool - that tells me it's more than likely an axe... does it weight negative cues?" And: "'can
smash a coconut' isn't very descriptive - could be a rock, a hammer, the back of an axe... But if you
know it can't, yet you also know that it can cut, those two combined tip you off that it's some kind
of blade. Such as a knife or sword."
**Mechanism (Zion: "why not store can't smash as... no_smash"):** a negative is a CONCEPT of its own.
`X can't Y` → X can `no-Y`, and `no-Y` is the opposite of `Y`; `it is hard to Y with X` → `hard-Y`,
partly-opposite. Nothing else is special: `no-smash` is shared, composed (`blade = sharp + cut +
no-smash`), asked about, rarity-weighted like any cue. The penalty is the OPPOSITE link doing its
work: a thing that does the opposite of one of your cues loses a flat point (partly: 0.6) — ruling
something out does not get weaker when many things could do it. Positive and negative weights differ
by construction: "can smash" is worth 1/log2(1+3) = 0.5 across hammer, rock and axe; "can't smash" is
a full point against each. And it is the same link hot/cold already use (Zion: "in most circumstances
it's very hard to have is hot and is cold; is blunt (smash) and sharp (no smash)") — a broth that is
hot is not ice, with no code written for temperature.
**Both at once (Zion: "if it's blunt and sharp, it's probably an axe of some kind - you can smash with
the back end while still chopping with the blade"):** hatchet can smash → "hammer or axe or rock?" →
can chop → "maybe a kind of axe?" — the one thing that does both. Kukri can cut, can't smash → "knife
or sword?". Same cues, opposite sign, different answer.
**The most specific kind that fits (`Brain.guess`):** candidates are scored without passing points up
to their kinds. One clear winner → "a kind of axe". Several tied (knife, sword, axe at 1.5) → what
they have in common → "some kind of tool"; tied with nothing in common → no guess, it says "maybe
knife or sword?". Compression again: the smallest description that still covers the evidence.
**Transcript:** tomahawk chops and cuts and has a handle → "maybe knife or sword or axe?" → "it is
hard to slice with a tomahawk" → "(I think a tomahawk is a kind of axe)". Kukri can smash → "maybe
hammer or axe or rock?" → can't smash a rock, can cut → "maybe a kind of knife or sword?".
**Honest notes:** "can't smash a rock" stores no object and no degree - just no-smash; absence of
a recorded no-X is not inability, so the FOR side only fires when someone said it. `X is hot` is
stored as "can hot" (a property is treated as an ability) - crude, but it keeps one mechanism.

### §26b. "Only": a constraint that carries identity

**Zion:** "More pressing is 'only'. Ice can only be cold; if it was warm, it's not ice. But rather water."
**Mechanism:** `X can only be Y` → X can Y AND X *only* Y. `only` is not a cue, it is a constraint:
in `intuit`, a candidate whose `only` is contradicted by the word's cues (anything that does an
opposite of Y) is not "less likely" - it is out, and the same the other way round. Being told `the ice is
warm` then does not add a cue: the thing has stopped being ice. What is it now? (Zion: "something that
is always cold logically must melt or evaporate when it warms up... It should be able to know that.")
Not told: DERIVED. Ice is a kind of water, and water can be warm - so it climbs the kinds and the
first one that admits the new state is what it became: "then it isn't ice - ice is only ever cold.
it must be water now: ice is water, and water can be warm." That inference is kept as an instance of
the `change` frame (before=ice, by=warm, after=water - §23a's before/by/after, worked out, not
narrated). `if X is Z, it is W` still exists for the cases it can't derive; with no kind that admits
the state it says it isn't ice and asks what warm ice is.
**Also fixed on the way:** `ice is water` (no article) is a KIND if the right-hand word is a thing it
knows (it has kinds, parts or a kind, and is not a property: no opposite, not a color/feeling/number),
else a property (`snow is white`). Properties are stored as `can` (Zion: "usually hot, but it can
also be warm... close enough to pass for now, until it accumulates enough to have proper grammar").
**Transcript:** "what is ice?" → "a kind of water (drink); it is only ever: cold" → "the ice is warm"
→ "then it isn't ice - ice is only ever cold. it must be water now: ice is water, and water can be
warm." → "what is ice?" → "...otherwise it's water". Nobody said melt.
**Honest notes:** `only` is binary; "ice is always cold" and "ice is usually cold" are one line apart
and should not be. `becomes` records one successor; melting/freezing/boiling as a cycle is three
such lines, not a cycle it sees.

### §26c. The name of a change it worked out

**Zion:** "Melting is going from solid to liquid." "Melting is transform."
**Mechanism:** `X is going from A to B` → a frame X (melt) with FIXED before/after fillers (solid,
liquid), a slot for `by`, is-a transform (= change in the seed). A change the brain derived (§26b:
ice > water, by warm) is then RECOGNIZED as melting if melt's fixed roles fit the instance's fillers:
ice is solid, water is liquid. "Fits" = is it, can be it, or is a kind of it (ice has two kinds here:
water and solid). Either order works: told melt first → the next warm ice says "that's melting"; told
melt after → "so ice becoming water - that was melting" (it re-reads what it already knows against
the new frame, §6's retroactive habit). `what did ice do?` → `melt: before=ice, by=warm, after=water`.
**What this is:** the event is known before its name (§26b), the name arrives as a frame over
properties, and the two meet by recognition - not by being told "ice melts". Freeze, evaporate and
condense are each one more line of the same shape; nothing is written for them.
**Honest notes:** the frame is matched on before/after only; `by` is unconstrained (melting by
"warm" vs. by "pressure" are the same to it). The -ing form is spelled by rule (melt → melting),
which will be wrong for some verbs until it hears them.

## §27. It learns grammar from exposure, not from rules (2026-10-22) — BUILT

**Zion:** "If the problem is that it can't understand proper grammar, make an adapter. If you heard
Japanese, even knowing how each word is pronounced, you might not speak it until you've heard words
strung into a sentence a few times and spoken it to someone who understands. Likewise you can speak
perfect Japanese and get along without knowing exactly what you're saying, then later through context
figure out the meaning."

The §24/§25 text layer parsed by fixed rules (a dozen regexes: strip a determiner, word 1 = subject,
word 2 = a known frame, fill the rest by position). It breaks on any shape nobody wrote a rule for.
`grammar.py` replaces the rules with the brain's OWN move from §23: line up a few example sentences,
and a position whose word VARIES becomes a slot, one whose word STAYS becomes a fixed anchor. Grammar
is frame-induction pointed at surface word order.

**Mechanism:** `teach(sentence, frame, fillers)` hears one example; the filler words mark the slots
(and their roles), determiners are absorbed into the slot they precede (so "ichigo gave…" and "the
knight gave…" are the SAME shape — determiners are optional, never a position of their own), and every
other word is a fixed anchor. Prepositions need no special case: "with" is simply a position that never
varies, so it survives as an anchor for free (`cut: [cutter] cut [thing] with [tool]`). A second example
of the same shape merges positionally (same honest limit as §23: alignment is positional only).

**Speak before you understand (the Japanese analogy, literally):** `parse()` binds every slot even when
the filler words mean nothing yet — the shape is spoken first. The words the brain has no concept for
are surfaced as exactly what to understand later, and resolving them is `intuit()` (§26), which places
a word from the company it keeps over a few uses. `produce(frame, fillers)` runs a pattern backwards to
say a meaning, and what it says, `parse()` reads back.

**Shown in `demo_grammar.py`:** two give-examples and two cut-examples → two learned patterns; "renji
gave byakuya a blade" (never seen) parses to give(giver=renji, receiver=byakuya, thing=blade); an
all-nonsense "zylo cut vorn with a florb" still binds every slot and surfaces {florb, vorn, zylo} as
not-yet-understood; produce(give, …) → "renji gave byakuya blade", which round-trips. `test_grammar.py`:
7 tests.

**Honest notes:** positional alignment only — no gaps, insertions or reordering (a real sequence
aligner is the same open piece §23 names); one content word per slot (multi-word fillers not handled);
verb tense rides as a literal anchor, so "gives/gave/giving" are three surfaces until each is heard or
aliased; produce() drops the article it absorbed ("byakuya blade", not "a blade"). The adapter uses the
brain only through its public methods; the core is untouched.

### §27a. Grammar with no labels at all (2026-10-23) — BUILT

Taking §27 the rest of the way (Zion: "take the adapter there next"). `observe(sentences)` induces
patterns from RAW sentences with no roles marked: strip the function words, bucket sentences that put
the same verb in the same place, and read the columns — a column that never varies is an anchor, one
that varies is a slot. Slots are auto-named s0/s1/… (as §23 names an unnarrated position step-k); the
verb names the frame. `name_slots(frame, {...})` attaches meaning afterwards — the shape is learned
first, understood later. The preposition "with" is discovered as an anchor unaided, because it is simply
the column that never varies. `demo_grammar_auto.py`; test_grammar.py now 12 tests.

**Honest notes:** grouping assumes English SVO with a one-word subject (the verb is taken to be the
second content word) — passives, questions, and subjectless imperatives are not induced; a slot that
happens to hold the same word across every example (a shared subject in a tiny corpus) is misread as an
anchor until more varied examples arrive; still one word per slot and positional alignment only (the
same open sequence-aligner §23 names). It is unsupervised shape induction, not yet unsupervised role
discovery — which slot is the giver still waits on narration or intuit().
