"""B.R.A.I.N. demo: dog / bear / wolf / cat on a synthetic perceptual substrate.

Shows, in order:
  1. one grounded sense -> everything similar is "dog ∘ descriptors"
  2. the SAME descriptor bundle recurring tips the compression rule -> a new concept splits off
  3. a one-legged dog stays "dog + 1-legged" (one descriptor never pays for a node)
  4. words bind to concepts by co-occurrence; the handle works both ways; an unlabeled kind still
     becomes a concept (handle ≠ sense)
  5. order-dependence: learn bear first and a dog is "bear + small + ..."
  6. idle(): unprompted consolidation makes a split nobody asked for
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_dog_bear.py
"""
import numpy as np
from brain import Brain, FEATURES

rng = np.random.default_rng(7)
#            size bulk legs snout paw diet tail fur claws whiskers
KINDS = {"dog": [4, 4, 4, 5, 3, 8, 6, 6, 1, 2],
         "wolf": [6, 5, 4, 7, 4, 10, 6, 7, 1, 2],
         "bear": [9, 9, 4, 6, 8, 5, 1, 7, 1, 1],
         "cat":  [2, 2, 4, 2, 2, 9, 7, 5, 9, 9]}


def sample(kind, **override):
    x = np.array(KINDS[kind], float) + rng.normal(0, 0.45, len(FEATURES))
    x[2] = 4.0
    for k, v in override.items(): x[FEATURES.index(k)] = v
    return x


def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------- 1-3: dog first
section("A. learn DOG first, then bears show up")
b = Brain()
for _ in range(5):
    b.learn(sample("dog"), "dog")
print("tree after 5 dogs:\n" + b.tree())

x = sample("bear")
print("\nfirst bear, explained with only 'dog' in the tree:")
print("  ->", b.describe(x), f"({b.tokens(x)} tokens)")
b.learn(x, "bear")
print("second bear:")
b.learn(sample("bear"), "bear")
print("tree:\n" + b.tree())
x = sample("bear")
print("\nthird bear now explained as:  ->", b.describe(x), f"({b.tokens(x)} tokens)")
b.learn(x, "bear")

section("B. a one-legged dog is still a dog")
x = sample("dog", leg_count=1)
print("  ->", b.describe(x), f"({b.tokens(x)} tokens)")
b.learn(x, "dog")
b.learn(sample("dog", leg_count=1), "dog")
dog = b.concept_for_word("dog")
_, core, stay, new = b.split_cost(dog, (("leg_count", -1),))
print(f"  compression check: cost(stay as dog + 1-legged) = {stay}   cost(new node) = {new}  -> no split")
print("tree:\n" + b.tree())

section("C. wolves: a few descriptors, then they earn a node")
for i in range(3):
    x = sample("wolf")
    print(f"  wolf {i+1}: ->", b.describe(x))
    b.learn(x, "wolf")
print("tree:\n" + b.tree())

section("D. handles: word -> concept, concept -> word, and an UNLABELED kind")
c = b.concept_for_word("bear")
print("  concept_for_word('bear') ->", c.label, "proto size/bulk/paw =", np.round(c.proto[[0, 1, 4]], 1))
x = sample("bear")
print("  unlabeled bear vector ->", b.describe(x))
for _ in range(3):                      # cats arrive with NO word at all
    b.learn(sample("cat"))
print("  cats fed with no word -> tree:\n" + b.tree())
print("  (a kind becomes a concept without a word; the handle is a door, not the sense)")
b.learn(sample("cat"), "cat"); b.learn(sample("cat"), "cat")
print("  after hearing 'cat' twice with cats -> ", b.describe(sample("cat")))

section("E. compression: what the tree bought")
test = [sample(k) for k in ("dog", "bear", "wolf", "cat") for _ in range(10)]
flat = Brain(log=lambda *a: None)
for _ in range(5): flat.learn(sample("dog"), "dog")
flat_tokens = sum(flat.tokens(x) for x in test)
tree_tokens = sum(b.tokens(x) for x in test)
print(f"  40 new animals described with only 'dog':   {flat_tokens} tokens")
print(f"  same 40 described with the grown tree:       {tree_tokens} tokens")
print(f"  tree has {len(b.nodes)} concepts; node overhead ~{3 * (len(b.nodes) - 1)} tokens, paid once, "
      f"then every future animal is ~1 token")

# ---------------------------------------------------------------- 5: order dependence
section("F. ORDER DEPENDENCE: learn BEAR first")
b2 = Brain()
for _ in range(5): b2.learn(sample("bear"), "bear")
x = sample("dog")
print("  a dog, with only 'bear' known:  ->", b2.describe(x))
b2.learn(x, "dog"); b2.learn(sample("dog"), "dog")
print("tree:\n" + b2.tree())
print("  (same world, different root: everything is defined relative to what came first)")

# ---------------------------------------------------------------- 6: idle thinking
section("G. idle(): consolidation nobody asked for")
b3 = Brain(log=lambda *a: None)
for _ in range(4): b3.learn(sample("dog"), "dog")
import brain as B
old = B.MIN_RECUR; B.MIN_RECUR = 99          # suppress online splits: just remember
for _ in range(3): b3.learn(sample("bear"), "bear")
for _ in range(3): b3.learn(sample("cat"), "cat")
B.MIN_RECUR = old
print("  tree while awake (splits held back):\n" + b3.tree())
b3.log = print
made = b3.idle()
print(f"  idle() made {made} split(s) on its own ->\n" + b3.tree())
