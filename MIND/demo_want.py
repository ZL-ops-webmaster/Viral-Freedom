"""B.R.A.I.N. demo 7: WANTING - born of knowing a thing exists, feeling attached to it, a fantasy of
having it, and a sense of what's possible.

  - a kid who never had a cookie, offered one, doesn't want it: doesn't know what it is
  - a kid who tasted one wants another
  - a kid who WATCHED Iron Man wants the suit - can't have it directly - the path runs through robotics
  - a want is what you return to; sleep can't take it; the memories attached to it survive
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_want.py
"""
import numpy as np
import brain as B
from brain import Brain

B.configure(
    ["sweet", "warm", "powered", "flying", "armored", "size", "edible", "made"],
    {"sweet": ("plain", "sweet"), "warm": ("cold", "warm"), "powered": ("unpowered", "powered"),
     "flying": ("grounded", "flying"), "armored": ("soft", "armored"), "size": ("small", "big"),
     "edible": ("inedible", "edible"), "made": ("natural", "made")})
from brain import FEATURES

rng = np.random.default_rng(4)
#                    sweet warm pow fly arm size edib made
KINDS = {"bread":     [2,   5,   0,  0,  0,  3,   9,   9],
         "cookie":    [9,   5,   0,  0,  0,  2,   9,   9],
         "armor":     [0,   3,   0,  0,  9,  6,   0,   9],
         "hero-suit": [0,   3,   9,  9,  9,  6,   0,   9]}
def sample(k): return np.array(KINDS[k], float) + rng.normal(0, 0.4, len(FEATURES))
def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

b = Brain(log=print)
for _ in range(4): b.learn(sample("bread"), "bread", reward=0.3)     # bread is fine. not a want.
for _ in range(3): b.learn(sample("armor"), "armor")                  # it knows what armor is

section("1. offered a cookie. it has never had one.")
x = sample("cookie")
print("  what it sees:", b.describe(x))
b.learn(x, "cookie")                                                  # offered, not tasted: no reward
print("  wants:", [w.name for w in b.desires()] or "nothing  <- doesn't know what a cookie IS to want it")

section("2. it tastes one")
b.learn(sample("cookie"), "cookie", reward=2.0)
print("  wants:", [f"{w.name} ({w.source})" for w in b.desires()])
print("  and now just SEEING one:")
b.learn(sample("cookie"), "cookie")                                   # no reward this time - just a cookie

section("3. it watches Iron Man (never wears the suit)")
for _ in range(3): b.watch(sample("hero-suit"), "hero-suit", reward=4.0)
b.think()
print("  wants:", [f"{w.name} ({w.source}, path: {w.path or 'obtainable'})" for w in b.desires()])

section("4. the return: what it thinks about when idle")
r = b.think()
print("  returned to:", r["returned_to"])
print("  the questions it cares about most:")
for h in r["questions"][:3]:
    print("   ", h["question"])
print("  (that is the kid who watched Iron Man getting into robotics)")

section("5. a dull week, then sleep: what survives")
for _ in range(15): b.learn(sample("bread"), "bread")
before = len(b.obs)
for _ in range(3): b.sleep()
print(f"  raw memories: {before} -> {len(b.obs)}")
print(f"  still remembers the suit: {len(b.recall('hero-suit'))}   the cookie: {len(b.recall('cookie'))}"
      f"   any given bread: {len(b.recall('bread'))}")
print("  wants carried through every night:", [w.name for w in b.desires()])
print("\n  'give me 3 years and I'd still remember where I'm at on exosuits' - because it's a want.")
