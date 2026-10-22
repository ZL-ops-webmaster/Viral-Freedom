"""B.R.A.I.N. demo 4: DOUBLES - one handle, several concepts (to / too / two; bat / bat).

  - a word is a door into a SET of nodes, not one
  - which sense is meant is resolved by CONTEXT: what is lit up right now (expectation, again)
  - the system NOTICES it holds a double, instead of silently picking the first
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_doubles.py
"""
import numpy as np
import brain as B
from brain import Brain

# one world that holds both bats: living things and made things, on shared dimensions
B.configure(
    ["size", "length", "weight", "hardness", "alive", "self_moving", "flies", "edge", "furred", "nocturnal"],
    {"size": ("small", "big"), "length": ("short", "long"), "weight": ("light", "heavy"),
     "hardness": ("soft", "hard"), "alive": ("inert", "alive"), "self_moving": ("still", "self-moving"),
     "flies": ("grounded", "flying"), "edge": ("blunt", "sharp"), "furred": ("feathered", "furred"),
     "nocturnal": ("daytime", "nocturnal")})
from brain import FEATURES

rng = np.random.default_rng(5)
#                   size len wt hard alive moves flies edge furred noct
KINDS = {"bird":     [2,  2,  1,  2,   9,    9,    9,   1,   0,    0],
         "bat_anim": [2,  2,  1,  2,   9,    9,    9,   1,   9,    9],   # the animal: furred, nocturnal bird-ish thing
         "ball":     [2,  2,  2,  6,   0,    0,    0,   0,   0,    0],
         "bat_club": [6,  8,  4,  9,   0,    0,    0,   1,   0,    0]}   # the club: big, long, heavy, hard

def sample(k): return np.array(KINDS[k], float) + rng.normal(0, 0.4, len(FEATURES))
def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

b = Brain(log=print)

section("1. two different things, both called 'bat' - learned in their natural company")
for _ in range(4):                       # birds and bats show up together (outdoors)
    b.learn(sample("bird"), "bird"); b.learn(sample("bat_anim"), "bat")
for _ in range(4):                       # balls and bats show up together (the game)
    b.learn(sample("ball"), "ball"); b.learn(sample("bat_club"), "bat")
b.think()
print("tree:\n" + b.tree())

section("2. it NOTICES the double")
def show(c): return f"{c.label} ({c.relation() or 'root'})"
for w, cands in b.doubles().items():
    print(f"  '{w}' opens {len(cands)} concepts: " + "  |  ".join(show(c) for c in cands))

section("3. 'bat' with nothing in mind -> honestly ambiguous")
b.recent = []
choice, cands = b.resolve("bat")
print("  resolve('bat') ->", choice.label if choice else f"AMBIGUOUS between {len(cands)} senses")

section("4. context resolves it (expectation, applied to a word)")
b.recent = []
b.learn(sample("ball"), "ball")                      # a ball is in mind...
choice, _ = b.resolve("bat")
print("  after seeing a ball,  'bat' ->", show(choice) if choice else "ambiguous")
b.recent = []
b.learn(sample("bird"), "bird")                      # ...now a bird is
choice, _ = b.resolve("bat")
print("  after seeing a bird,  'bat' ->", show(choice) if choice else "ambiguous")

section("5. the same thing, as to / too / two")
print("  spoken /tu:/ is one handle over three concepts; the sentence around it (what's active) picks one.")
print("  the WRITTEN handle differs - two / to / too - so the written channel is what teaches the split:")
print("  hear the same sound with three different spellings, and the sound binds to three nodes, not one.")
print("  (that is why the child has to learn to spell them: spelling is disambiguation)")
