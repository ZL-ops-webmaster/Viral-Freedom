"""B.R.A.I.N. demo 12: A WEB, NOT A TREE.

  "Fish and sharks are vertebrates, but so is every mammal. 'Shark' refers to great white sharks,
   shark teeth, but also whale sharks - which also connect to whales despite the differences."
  - the tree keeps ONE definitional parent per concept (that is what makes descriptions short)
  - a generic word that cuts ACROSS branches ('pet') becomes a membership, not a parent
  - a compound of two concept handles ('whale shark') links both: is-a shark, linked to whale
  - a handle is a hub: web('shark') fans out to kinds, memberships, links, and mentions
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_web.py
"""
import numpy as np
import brain as B
from brain import Brain

B.configure(
    ["size", "legs", "fur", "gills", "aquatic", "sharp_teeth", "filter_feeds", "whiskers", "warm_blood", "bones"],
    {"size": ("small", "big"), "legs": ("legless", "legged"), "fur": ("hairless", "furred"),
     "gills": ("lunged", "gilled"), "aquatic": ("land", "aquatic"), "sharp_teeth": ("blunt-teeth", "sharp-teeth"),
     "filter_feeds": ("hunts", "filter-feeds"), "whiskers": ("no-whiskers", "whiskers"),
     "warm_blood": ("cold-blooded", "warm-blooded"), "bones": ("boneless", "bony")})
from brain import FEATURES

rng = np.random.default_rng(6)
KINDS = {"dog":         [4, 4, 8, 0, 0, 6, 0, 3, 9, 9],
         "cat":         [2, 4, 7, 0, 0, 6, 0, 9, 9, 9],
         "whale":       [10, 0, 0, 0, 9, 2, 8, 0, 9, 9],
         "goldfish":    [1, 0, 0, 9, 9, 1, 0, 0, 1, 9],
         "shark":       [8, 0, 0, 9, 9, 9, 0, 0, 1, 4],
         "jellyfish":   [2, 0, 0, 0, 9, 0, 0, 0, 1, 0],
         "whale shark": [10, 0, 0, 9, 9, 1, 9, 0, 1, 4]}
def sample(k): return np.clip(np.array(KINDS[k], float) + rng.normal(0, 0.4, len(FEATURES)), 0, 10)
def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
def show_web(b, w):
    d = b.web(w)
    print(f"  '{w}':  is {' > '.join(d['is']) or '-'}   |   kinds: {', '.join(d['kinds']) or '-'}   |   "
          f"linked to: {', '.join(d['linked']) or '-'}   |   mentioned in: {', '.join(d['mentioned_in']) or '-'}")

b = Brain(log=print)
for k in ("dog", "cat", "whale", "goldfish", "shark", "jellyfish"):
    for _ in range(3): b.learn(sample(k), k)

section("1. generic words build the tree where they can")
for k in ("dog", "cat", "whale"):
    for _ in range(2): b.learn(sample(k), "mammal")
for k in ("goldfish", "shark"):
    for _ in range(2): b.learn(sample(k), "fish")
for k in ("dog", "cat", "whale", "goldfish", "shark", "jellyfish"):
    for _ in range(2): b.learn(sample(k), "animal")
for k in ("dog", "cat", "whale", "goldfish", "shark"):           # not the jellyfish
    for _ in range(2): b.learn(sample(k), "vertebrate")
print("  (words bind during the day; the tree reorganises when it idles)")
b.think()
print("tree:\n" + b.tree())

section("2. ...and become memberships where they can't: 'pet' cuts across mammal and fish")
for k in ("dog", "cat", "goldfish"):
    for _ in range(2): b.learn(sample(k), "pet")
b.think()
show_web(b, "dog"); show_web(b, "goldfish")

section("3. 'whale shark' - a compound of two things it knows")
b.hear("whale shark")
for _ in range(2): b.learn(sample("whale shark"))                  # no word: it names it from what it heard
print("tree (sharks):")
print("\n".join(l for l in b.tree().splitlines() if "shark" in l))

section("4. the web, from a handle")
for w in ("shark", "whale", "whale shark", "jellyfish"):
    show_web(b, w)
print("\n  one definitional parent each (short descriptions); everything else is a link.")
