"""B.R.A.I.N. demo 10: THE WORD IS THE TRIGGER, and the old link is kept.

  "I hear bear, I see big dog, I see the difference - paw size, shoulder width, snout - and I make a
   new node. Bear. So when I hear 'fire bear' I don't think 'big dog wide paws different snout covered
   in fire', I think 'bear covered in fire'.
   But I still remember that I used to connect the two - which lets me approximate a sense of humour:
   showing someone a picture of a bear and saying 'I want this kind of dog'."
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_bear.py
"""
import numpy as np
import brain as B
from brain import Brain

B.configure(
    ["size", "bulk", "leg_count", "snout_len", "paw_width", "diet", "tail_len", "fur", "claws_retract", "whiskers", "glow"],
    {"size": ("small", "big"), "bulk": ("slender", "broad"), "snout_len": ("short-snout", "long-snout"),
     "paw_width": ("narrow-paws", "wide-paws"), "diet": ("plant-leaning-diet", "meat-leaning-diet"),
     "tail_len": ("short-tail", "long-tail"), "fur": ("thin-fur", "thick-fur"),
     "claws_retract": ("fixed-claws", "retractable-claws"), "whiskers": ("no-whiskers", "whiskers"),
     "glow": ("dark", "on-fire")})
from brain import FEATURES

rng = np.random.default_rng(2)
KINDS = {"dog":  [4, 4, 4, 5, 3, 8, 6, 6, 1, 2, 0],
         "bear": [9, 9, 4, 6, 8, 5, 1, 7, 1, 1, 0]}
def sample(k, **o):
    x = np.array(KINDS[k], float) + rng.normal(0, 0.45, len(FEATURES)); x[2] = 4
    for f, v in o.items(): x[FEATURES.index(f)] = v
    return x
def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

b = Brain(log=print)
for _ in range(4): b.learn(sample("dog"), "dog")

section("1. hears 'bear' once, sees one")
x = sample("bear")
print("  what it sees before the word lands:", b.describe(x))
b.learn(x, "bear")
print("tree:\n" + b.tree())

section("2. 'fire bear'")
fb = sample("bear", glow=9)
root = b.concept_for_word("dog")
long_form = "dog + " + " + ".join(B.dev_word(d, fb[FEATURES.index(d)], root.proto[FEATURES.index(d)]) for d, _ in root.deviations(fb))
print("  without the bear node it would have to say:", long_form)
print("  with it:                                   ", b.describe(fb))

section("3. but the link isn't gone")
bear = b.concept_for_word("bear")
print(" ", b.used_to_think(bear))

section("4. ...which is how you make a joke")
print(f"  *shows a picture of a bear*   '{b.misread(bear)}'  -> 'I want {b.misread(bear)}'")
print("  (only works because both levels and the link between them are still there)")
