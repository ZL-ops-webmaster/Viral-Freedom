"""B.R.A.I.N. demo 3: ACCUMULATION - the more it knows, the cheaper the next thing is.

A child who knows swords hears "broadsword", sees one picture, and has it instantly:
  - the WORD composes: broad + sword  -> he expects it before he sees it
  - the first thing he notices is the DEVIATION: "this sword is wider"
  - "broad" was learned somewhere else entirely and transfers
Same engine as the animals; only the world it senses changes (configure()).
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_broadsword.py
"""
import numpy as np
import brain as B
from brain import Brain

# a different world: medieval weapons. The mechanism doesn't change; what it can sense does.
B.configure(
    ["length", "width", "weight", "edge", "point", "grip", "curve", "ornament"],
    {"length": ("short", "long"), "width": ("narrow", "broad"), "weight": ("light", "heavy"),
     "edge": ("blunt-edge", "sharp-edge"), "point": ("blunt-point", "sharp-point"),
     "grip": ("short-grip", "long-grip"), "curve": ("straight", "curved"), "ornament": ("plain", "ornate")})
from brain import FEATURES

rng = np.random.default_rng(3)
#                length width weight edge point grip curve ornament
KINDS = {"sword":      [7, 2,   5,    8,   8,    3,   1,    3],
         "dagger":     [3, 1.5, 2,    8,   9,    2,   1,    3],
         "axe":        [5, 6,   8,    7,   2,    4,   3,    2],
         "broadsword": [7, 4.5, 6,    8,   7,    3,   1,    3]}


def sample(kind): return np.array(KINDS[kind], float) + rng.normal(0, 0.4, len(FEATURES))
def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


b = Brain(log=print)

section("1. a child who is really into medieval weapons")
for _ in range(4): b.learn(sample("sword"), "sword")
for _ in range(3): b.learn(sample("dagger"), "dagger")
for _ in range(3): b.learn(sample("axe"), "axe")
b.think()
print("tree:\n" + b.tree())
print("\ndescriptor concepts it has now (minted from its own splits):")
for name, d in b.descriptor_concepts().items():
    print(f"  {name:14s} = {d['dim']} {d['delta']:+.1f}   from: {', '.join(d['from'])}")

section("2. he hears a NEW WORD: 'broadsword'. No picture yet.")
h = b.hear("broadsword")
print(f"  expects, before seeing one:  width ~{h['proto'][1]:.1f}  (a sword is ~{b.concept_for_word('sword').proto[1]:.1f})")

section("3. one picture. no label given with it.")
x = sample("broadsword")
sword = b.concept_for_word("sword")
first = [B.dev_word(d, x[FEATURES.index(d)], sword.proto[FEATURES.index(d)]) for d, _ in sword.deviations(x)]
print(f"  the first thing it notices vs a sword: {', '.join(first) or 'nothing'}")
out = b.learn(x)                                   # <- no word passed; it names it from what it HEARD
print(f"  learn() -> {out}")
print("tree:\n" + b.tree())
print("  a second one:", b.describe(sample("broadsword")))

section("4. ACCUMULATION: evidence needed to learn each concept, in order")
print(f"  {'concept':12s} {'observations it took to exist':>32s}")
for c in b.nodes:
    how = "one picture + the word" if c.born_from == 1 and c.parent else ("the first grounded sense" if c.parent is None else "recurrence")
    print(f"  {c.label:12s} {c.born_from:>8d}   {how}")
print("\n  what this shows (no more than this): concepts learned by RECURRENCE cost 2-3 observations;")
print("  the one learned by COMPOSITION cost 1 observation + a word it already understood.")
print("  that drop is the compounding: the richer the tree, the more new things are just old parts.")
