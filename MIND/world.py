"""A procedurally generated hierarchical world, so accumulation can actually be tested.

Not hand-authored: base kinds get random profiles, sub-kinds are a base plus a few descriptor deltas,
sub-sub-kinds go one deeper. The generator keeps the TRUE hierarchy, so a learned tree can be scored
against it instead of eyeballed. Nothing here is tuned to a story.
"""
import numpy as np

F = 12
FEATURES = [f"f{i}" for i in range(F)]
DESCRIPTOR_WORDS = {f: (f"low-{f}", f"high-{f}") for f in FEATURES}


def make_world(seed=0, bases=6, subs=3, subsubs=(0, 2), spread=3.5):
    rng = np.random.default_rng(seed)
    kinds = {}                      # name -> profile
    parent = {}                     # name -> true parent name (None for bases)
    # bases: spread apart so they are genuinely different things
    profiles = []
    while len(profiles) < bases:
        p = rng.uniform(1, 9, F)
        if all(np.abs(p - q).max() > 4 for q in profiles):
            profiles.append(p)
    for i, p in enumerate(profiles):
        b = f"K{i+1}"; kinds[b] = p; parent[b] = None
        for j in range(subs):
            s = f"{b}.{j+1}"; d = rng.choice(F, size=rng.integers(2, 5), replace=False)
            q = p.copy(); q[d] = np.clip(q[d] + rng.choice([-1, 1], size=len(d)) * spread, 0, 10)
            kinds[s] = q; parent[s] = b
            for k in range(rng.integers(subsubs[0], subsubs[1] + 1)):
                t = f"{s}.{k+1}"; d2 = rng.choice(F, size=rng.integers(1, 4), replace=False)
                r = q.copy(); r[d2] = np.clip(r[d2] + rng.choice([-1, 1], size=len(d2)) * spread, 0, 10)
                kinds[t] = r; parent[t] = s
    return kinds, parent


def curriculum(kinds, n, seed=0, noise=0.45):
    """A long, mixed stream of observations with words - the way a life arrives."""
    rng = np.random.default_rng(seed + 1)
    names = list(kinds)
    depth = np.array([name.count(".") for name in names])
    w = 1.0 / (1 + depth)            # common things are common; deep things rarer
    w /= w.sum()
    for _ in range(n):
        k = names[rng.choice(len(names), p=w)]
        yield np.clip(kinds[k] + rng.normal(0, noise, F), 0, 10), k


def score(brain, kinds, parent, samples=400, seed=99):
    """How much of the true structure did it recover? Purity: the majority true kind among the
    observations each learned concept claims. Recovery: true kinds that have a concept bound to their
    word. Depth: learned vs true."""
    rng = np.random.default_rng(seed)
    names = list(kinds)
    claims = {}
    for _ in range(samples):
        k = names[rng.integers(len(names))]
        x = np.clip(kinds[k] + rng.normal(0, 0.45, F), 0, 10)
        c = brain.nearest(x)
        claims.setdefault(c.cid, []).append(k)
    purity = np.mean([max(np.unique(v, return_counts=True)[1]) / len(v) for v in claims.values()])
    recovered = sum(1 for k in kinds if brain.concepts_for_word(k))
    def depth(c): return 0 if c.parent is None else 1 + depth(c.parent)
    learned_depth = max(depth(c) for c in brain.nodes)
    true_depth = max(k.count(".") for k in kinds)
    return {"concepts": len(brain.nodes), "true_kinds": len(kinds), "recovered": recovered,
            "purity": round(float(purity), 3), "learned_depth": learned_depth, "true_depth": true_depth}
