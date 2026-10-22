"""B.R.A.I.N. demo 9: ACCUMULATION AT SCALE - does the tree hold up when the world is big?

Every earlier demo was a toy of 4-5 concepts. This feeds hundreds of observations from a procedurally
generated world of ~40 kinds in a real hierarchy, with days and nights, and SCORES the learned tree
against the generator's ground truth. Nothing is tuned to a story.
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_accumulate.py
"""
import time
import numpy as np
import brain as B
import world as W

B.configure(W.FEATURES, W.DESCRIPTOR_WORDS)
from brain import Brain

kinds, parent = W.make_world(seed=1)
print(f"world: {len(kinds)} kinds, depth {max(k.count('.') for k in kinds)}, "
      f"{sum(1 for p in parent.values() if p is None)} bases")

b = Brain(log=lambda *a: None)
N, DAY = 600, 100
t0 = time.time()
splits_log = []
b.log = lambda *a: splits_log.append(" ".join(map(str, a)))
for i, (x, word) in enumerate(W.curriculum(kinds, N, seed=1), 1):
    b.learn(x, word)
    if i % DAY == 0:
        b.think(); b.sleep()
        s = W.score(b, kinds, parent)
        print(f"  day {i//DAY}: {s['concepts']:3d} concepts  recovered {s['recovered']:2d}/{s['true_kinds']}  "
              f"purity {s['purity']:.2f}  depth {s['learned_depth']}/{s['true_depth']}  "
              f"raw memories {len(b.obs):3d}  ({time.time()-t0:.1f}s)")
b.think()

print("\nlearned tree (first 30 lines):")
print("\n".join(b.tree().splitlines()[:30]))

# compression at scale: fresh samples described with the tree vs with the root alone
rng = np.random.default_rng(7)
names = list(kinds)
fresh = [np.clip(kinds[names[rng.integers(len(names))]] + rng.normal(0, 0.45, W.F), 0, 10) for _ in range(200)]
root = [c for c in b.nodes if c.parent is None][0]
flat = sum(1 + len(root.deviations(x)) for x in fresh)
tree = sum(b.tokens(x) for x in fresh)
print(f"\n200 fresh things: {flat} tokens with one concept, {tree} with the tree  "
      f"(node overhead paid once: ~{3*(len(b.nodes)-1)})")
print(f"splits that fired: {sum('SPLIT' in l for l in splits_log)}   questions open now: {len(b.hypotheses)}")
print(f"\nhonest read: purity says how clean each concept is; recovered says how many true kinds got a word; "
      f"depth says whether it found the hierarchy or flattened it.")
