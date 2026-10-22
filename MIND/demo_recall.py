"""B.R.A.I.N. demo 5: RETROACTIVE MEMORY - the past is re-read through the present.

  I see a bat. Never heard the word. It flies; birds fly. I call it a bird.
  Later someone says "that's a bat." I learn it.
  Someone asks: have you ever seen a bat?  Yes.
  (and the number of birds I've seen just went down.)
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_recall.py
"""
import numpy as np
import brain as B
from brain import Brain

B.configure(
    ["size", "length", "weight", "hardness", "alive", "self_moving", "flies", "edge", "furred", "nocturnal"],
    {"size": ("small", "big"), "length": ("short", "long"), "weight": ("light", "heavy"),
     "hardness": ("soft", "hard"), "alive": ("inert", "alive"), "self_moving": ("still", "self-moving"),
     "flies": ("grounded", "flying"), "edge": ("blunt", "sharp"), "furred": ("feathered", "furred"),
     "nocturnal": ("daytime", "nocturnal")})
from brain import FEATURES

rng = np.random.default_rng(9)
KINDS = {"bird": [2, 2, 1, 2, 9, 9, 9, 1, 0, 0],
         "bat":  [2, 2, 1, 2, 9, 9, 9, 1, 9, 9]}
def sample(k): return np.array(KINDS[k], float) + rng.normal(0, 0.4, len(FEATURES))
def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

b = Brain(log=print)

section("1. it knows birds")
for _ in range(4): b.learn(sample("bird"), "bird")
print("  birds seen so far:", len(b.recall("bird")))

section("2. a bat flies past. it has never heard the word. it flies, birds fly...")
for i in range(3):
    print(f"  sighting {i+1}: it calls this ->", b.learn(sample("bat")))     # NO word given
print("tree:\n" + b.tree())
print("  'have you seen a bat?'  ->", "yes" if b.recall("bat") else "no, I don't know that word")

section("3. someone corrects it: 'that's a bat'")
b.learn(sample("bat"), "bat"); b.learn(sample("bat"), "bat")
print("tree:\n" + b.tree())

section("4. 'have you ever seen a bat?'")
hits = b.recall("bat")
print(f"  yes - {len(hits)} times.")
for i, then, now in hits:
    print(f"    observation #{i}: I called it '{then}' at the time; I read it as '{now}' now")

section("5. and the other direction: 'how many birds have you seen?'")
print(f"  {len(b.recall('bird'))} - the three it mistook for birds are no longer counted as birds.")
print("  nothing was rewritten. the sightings were stored raw; only the reading changed.")
