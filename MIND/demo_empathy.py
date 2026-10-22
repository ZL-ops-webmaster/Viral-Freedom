"""B.R.A.I.N. demo 8: EMPATHY - "a kid helps another kid who fell, because they fell themselves and
know it can hurt."

  - feeling bound by LIVING it fires again on recognition, at full strength
  - no pain reached me, so it's happening to someone else -> the want points outward: make it stop
  - the kid who never fell feels nothing, and walks past
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_empathy.py
"""
import numpy as np
import brain as B
from brain import Brain

B.configure(
    ["running", "fell", "on_ground", "crying", "speed", "height"],
    {"running": ("still", "running"), "fell": ("upright", "fallen"), "on_ground": ("standing", "down"),
     "crying": ("quiet", "crying"), "speed": ("slow", "fast"), "height": ("low", "high")})
from brain import FEATURES

rng = np.random.default_rng(8)
KINDS = {"playing": [8, 0, 0, 0, 7, 2],
         "falling": [3, 9, 9, 7, 4, 1]}
def sample(k): return np.array(KINDS[k], float) + rng.normal(0, 0.4, len(FEATURES))
def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

kid_a = Brain(log=lambda *a: print("   [A]", *a))
kid_b = Brain(log=lambda *a: print("   [B]", *a))
for k in (kid_a, kid_b):
    for _ in range(4): k.learn(sample("playing"), "playing", reward=0.5)

section("1. kid A falls. it hurts. kid B never has.")
for _ in range(2): kid_a.learn(sample("falling"), "falling", reward=-2.0)
for _ in range(2): kid_b.watch(sample("falling"), "falling")        # B has SEEN falling, never felt it
print("  A's feeling about falling:", f"{kid_a.valence_of(kid_a.concept_for_word('falling')):+.1f}")
print("  B's feeling about falling:", f"{kid_b.valence_of(kid_b.concept_for_word('falling')):+.1f}")

section("2. another kid falls in front of both of them (no pain reaches either)")
x = sample("falling")
print("  A:"); kid_a.learn(x, "falling")
print("  B:"); kid_b.learn(x, "falling")
print()
print("  A wants:", [w.name for w in kid_a.desires()] or "nothing")
print("  B wants:", [w.name for w in kid_b.desires()] or "nothing   <- walks past. doesn't know it hurts.")

section("3. then B falls, once. and the next time it sees it happen:")
kid_b.learn(sample("falling"), "falling", reward=-2.0)
print("  B:"); kid_b.learn(sample("falling"), "falling")
print("  B wants:", [w.name for w in kid_b.desires()])
print("\n  you help because you know. the feeling was earned, so it fires for someone else at full strength.")
