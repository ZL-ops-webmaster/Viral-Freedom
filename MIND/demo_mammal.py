"""B.R.A.I.N. demo 11: RETROACTIVELY ADDING TREES.

  "Dogs and cats are both mammals, and animals - but fish are also animals, without being mammals."
  A generic word, applied to several kinds, inserts a parent ABOVE them. The historical chain (cat
  filed under dog) dissolves the moment a word gives it a reason to. Compression keeps 'bat' from
  doing the same thing to the animal and the club.
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_mammal.py
"""
import numpy as np
import brain as B
from brain import Brain

B.configure(
    ["size", "bulk", "leg_count", "snout_len", "paw_width", "diet", "tail_len", "fur", "claws_retract", "whiskers"],
    {"size": ("small", "big"), "bulk": ("slender", "broad"), "leg_count": ("fewer-legs", "more-legs"),
     "snout_len": ("short-snout", "long-snout"), "paw_width": ("narrow-paws", "wide-paws"),
     "diet": ("plant-leaning-diet", "meat-leaning-diet"), "tail_len": ("short-tail", "long-tail"),
     "fur": ("thin-fur", "thick-fur"), "claws_retract": ("fixed-claws", "retractable-claws"),
     "whiskers": ("no-whiskers", "whiskers")})
from brain import FEATURES

rng = np.random.default_rng(12)
KINDS = {"dog":  [4, 4, 4, 5, 3, 8, 6, 6, 1, 2],
         "cat":  [2, 2, 4, 2, 2, 9, 7, 5, 9, 9],
         "fish": [3, 2, 0, 3, 0, 7, 8, 0, 0, 0]}
def sample(k): return np.array(KINDS[k], float) + rng.normal(0, 0.4, len(FEATURES))
def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

b = Brain(log=print)
for _ in range(4): b.learn(sample("dog"), "dog")
for _ in range(3): b.learn(sample("cat"), "cat")
print("tree, the way it was learned (cat filed under dog - the historical accident):\n" + b.tree())

section("1. 'mammal' - said about a dog, then about a cat")
b.learn(sample("dog"), "mammal"); b.learn(sample("dog"), "mammal")
b.learn(sample("cat"), "mammal"); b.learn(sample("cat"), "mammal")
b.think()                                                  # the tree reorganises when it idles
print("tree:\n" + b.tree())
print("  ", b.used_to_think(b.concept_for_word("cat")))

section("2. a fish shows up (gets filed under whatever's nearest, like everything does)")
for _ in range(3): b.learn(sample("fish"), "fish")
print("tree:\n" + b.tree())

section("3. 'animal' - said about a dog, a cat, and a fish")
for k in ("dog", "cat", "fish"):
    for _ in range(2): b.learn(sample(k), "animal")
b.think()
print("tree:\n" + b.tree())
print("  ", b.used_to_think(b.concept_for_word("fish")))
print("\n  fish is an animal without being a mammal. dog and cat are siblings now, not a chain.")
