"""B.R.A.I.N. demo 2: THINKING and FEELING on top of learning (design.md §5, §5b).

  learning  = world -> tree           (demo_dog_bear.py)
  thinking  = tree -> prediction / imagination / question, with NO input
  feeling   = the derivative of compression: it decides which thought comes next
  the loop  = think -> ask -> the world answers -> learn -> feel
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_think.py
"""
import numpy as np
import brain as B
from brain import Brain, FEATURES

rng = np.random.default_rng(11)
#             size bulk legs snout paw diet tail fur claws whiskers
KINDS = {"dog":  [4, 4, 4, 5, 3, 8, 6, 6, 1, 2], "wolf": [6, 5, 4, 7, 4, 10, 6, 7, 1, 2],
         "bear": [9, 9, 4, 6, 8, 5, 1, 7, 1, 1], "cat":  [2, 2, 4, 2, 2, 9, 7, 5, 9, 9],
         "lion": [7, 6, 4, 3, 5, 9, 7, 5, 9, 9]}   # lion = a big, broad, wide-pawed cat


def sample(kind):
    x = np.array(KINDS[kind], float) + rng.normal(0, 0.45, len(FEATURES)); x[2] = 4.0; return x


def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


journal = []                                         # everything it logs, so the summary is TRUE
def jlog(*a):
    s = " ".join(str(t) for t in a); journal.append(s); print(s)

b = Brain(log=lambda *a: None)
for _ in range(5): b.learn(sample("dog"), "dog")

# ---------------------------------------------------------------- FEELING
section("1. FEELING: unease is compression debt; satisfaction is compression gain")
old = B.MIN_RECUR; B.MIN_RECUR = 99                 # hold splits back: just pile up the unexplained
for k in ("bear", "wolf", "cat"):
    for _ in range(3): b.learn(sample(k), k)
B.MIN_RECUR = old
a, mood = b.feel()
print("  after 9 animals it can't explain  ->  feels", mood)
print("  tree:\n" + b.tree())

b.log = jlog
print("\n  think()  (nothing asked for this):")
r = b.think()
print(f"  mood before: {r['mood_before']}")
print(f"  consolidated {r['consolidated']} concept(s) on its own")
print(f"  mood after:  {r['mood_after']}")
print("  tree:\n" + b.tree())

# ---------------------------------------------------------------- THINKING
section("2. descriptor concepts, MINTED from the splits (the lookup table is gone)")
for name, d in b.descriptor_concepts().items():
    print(f"  {name:20s} = {d['dim']} {d['delta']:+.1f}   learned from: {', '.join(d['from'])}")

section("3. PREDICTION: fill in what it hasn't seen, from the structure")
partial = np.array([9, 9] + [np.nan] * (len(FEATURES) - 2))
c, filled, fit = b.predict(partial)
print("  given only size=9, bulk=9  ->  expects:", c.label)
print("  and predicts, before seeing them:  paw_width=%.1f  diet=%.1f  tail_len=%.1f"
      % (filled[4], filled[5], filled[6]), " (bear: wide paws, plant-leaning, short tail)")

section("4. IMAGINATION / ANALOGY: apply 'big' (learned from dogs) to a CAT")
p, near, dist = b.imagine(b.concept_for_word("cat"), ["big"])
print(f"  imagined 'big cat': size {p[0]:.1f}, bulk {p[1]:.1f}  ->  nearest known concept: {near.label}, "
      f"distance {dist:.1f}  ->  {'NOVEL: nothing it knows covers this' if dist > b.NOVEL else 'already known'}")

section("5. CURIOSITY: the gaps in the tree become questions (unprompted)")
for h in b.wonder(top=6):
    print(f"  {h['question']:28s} novelty {h['novelty']:.1f}")
a, mood = b.feel()
print("  feels ->", mood)
open_q = [h["question"] for h in b.hypotheses]
print(f"  open questions held in mind: {len(open_q)};  'is there a big cat?' among them: "
      f"{'is there a big cat?' in open_q}")

section("6. THE LOOP: the world answers a question it asked")
print("  a lion walks in (it has never seen one). bottom-up it looks like a weird bear;")
print("  the question it is already holding says otherwise:")
for i in range(3):
    x = sample("lion")
    near = b.nearest(x)
    out = b.learn(x, "lion")
    print(f"   lion {i+1}: nearest by distance alone = {near.label};  learn() -> {out}")
a, mood = b.feel()
print("  feels ->", mood)
print("  tree:\n" + b.tree())

section("7. what actually happened (from its own journal, not a script)")
for line in journal:
    if any(k in line for k in ("SPLIT", "ANSWERED")):
        print(" ", line.strip())
d = b.descriptor_concepts()
print(f"  'big' is now a concept learned from: {', '.join(d['big']['from'])}")
