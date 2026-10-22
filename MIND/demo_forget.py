"""B.R.A.I.N. demo 6: FORGETTING - priority, not time.

  "Give me 3 years and I'd still remember where I'm at on exosuits. But ask me the second I get home
   whether a washer broke at work and I might not know. Big questions I'll forget I asked - then
   idling, I'll wonder them again."
  - the event goes, the concept stays
  - what survives a night is what mattered: surprising, significant, re-read, central
  - questions are never stored; the ones that still matter are regenerated from the tree's gaps
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_forget.py
"""
import numpy as np
from brain import Brain, FEATURES

rng = np.random.default_rng(21)
KINDS = {"dog":  [4, 4, 4, 5, 3, 8, 6, 6, 1, 2], "bear": [9, 9, 4, 6, 8, 5, 1, 7, 1, 1],
         "cat":  [2, 2, 4, 2, 2, 9, 7, 5, 9, 9], "lion": [7, 6, 4, 3, 5, 9, 7, 5, 9, 9]}
def sample(k): x = np.array(KINDS[k], float) + rng.normal(0, 0.45, len(FEATURES)); x[2] = 4; return x
def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

b = Brain(log=print)
for _ in range(4): b.learn(sample("dog"), "dog")
for _ in range(3): b.learn(sample("bear"), "bear")
for _ in range(3): b.learn(sample("cat"), "cat")
b.think()

section("1. a day: twenty ordinary dogs (the washer at work), and one lion (the thing that mattered)")
for _ in range(20): b.learn(sample("dog"), "dog")           # fit perfectly, change nothing
b.think()                                                     # it wonders; 'is there a big cat?' is in mind
b.learn(sample("lion"), "lion")                               # answers a question -> a concept is minted
dog = b.concept_for_word("dog")
print(f"  raw memories held: {len(b.obs)}   open questions: {len(b.hypotheses)}   dog concept: n={dog.n}")
print(f"  dogs it can recall seeing today: {len(b.recall('dog'))}    lions: {len(b.recall('lion'))}")

section("2. a night")
b.sleep()
print(f"  raw memories held: {len(b.obs)}   open questions: {len(b.hypotheses)}   dog concept: n={dog.n}  <- untouched")
print(f"  dogs it can recall seeing: {len(b.recall('dog'))}   (it still KNOWS dogs; it doesn't remember each one)")
print(f"  the lion: {len(b.recall('lion'))}   ('I called it \"{b.recall('lion')[0][1]}\" at the time')")

section("3. a big question it dropped - then wonders again when idle")
r = b.think()
print(f"  questions regenerated from the tree's gaps, unprompted: {len(b.hypotheses)}")
for h in r["questions"][:3]:
    print("   ", h["question"])

section("4. three more nights: what survives")
for _ in range(3): b.sleep()
print(f"  raw memories still held: {len(b.obs)}  ->", [s for _, s, _ in b.recall('lion')] or "lion forgotten too")
print(f"  concepts: {len(b.nodes)}  (none lost)    tree:\n" + b.tree())
print("\n  what it kept after four nights is what it keeps for years: the concepts, and the day a lion walked in.")
