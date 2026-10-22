"""B.R.A.I.N. demo 15: A FRAME FROM EXPERIENCE - a recurring sequence whose fillers vary.

  "Every zanpakuto goes through basic sword > command > transform > new sword, when shikai is used.
   Bankai is explained as a progression of shikai."
  - episode(): one experience, in order; after two, varying positions become ROLES, constant ones
    become fixed parts, and the narration's labels name them
  - every filler seen at a position is a connection -> fit() comes from the same data
  - a sequence that begins with another's and goes on is a PROGRESSION of it
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_shikai.py
"""
from brain import Brain

def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
b = Brain(log=print)
for sealed in ("zangetsu-sealed", "sode-sealed", "zabimaru-sealed"):   # it knows these are all swords
    b._ensure(sealed); b._ensure("sword"); b.structs[sealed].links.append(("is-a", "sword"))

section("1. shikai, watched three times (different characters, different swords, different becomings)")
LAB = ["weapon", "command", "transform", "new-form"]
b.episode("shikai", ["zangetsu-sealed", "release", "blade-widens-in-light", "zangetsu-cleaver"], LAB, who="ichigo")
b.episode("shikai", ["sode-sealed", "dance", "blade-turns-white-with-frost", "sode-white"], LAB, who="rukia")
b.episode("shikai", ["zabimaru-sealed", "roar", "blade-segments-apart", "zabimaru-whip"], LAB, who="renji")
print()
print(b.breakdown("shikai"))

section("2. 'bankai is a progression of shikai': the same steps, then more")
LAB2 = LAB + ["bankai-command", "bankai-transform", "final-form"]
b.episode("bankai", ["zangetsu-sealed", "release", "blade-widens-in-light", "zangetsu-cleaver",
                     "bankai", "burst-of-black-reiatsu", "tensa-zangetsu"], LAB2, who="ichigo")
b.episode("bankai", ["senbonzakura-sealed", "scatter", "blade-dissolves-into-petals", "senbonzakura-petals",
                     "bankai", "thousand-blades-rise-from-the-ground", "senbonzakura-kageyoshi"], LAB2, who="byakuya")
print()
print(b.breakdown("bankai"))

section("2b. the before, the after, AND the becoming - each kept, each its own thing")
print(b.breakdown("ichigo's bankai"))
print()
print(b.breakdown("ichigo's bankai-transform"))
print()
print(b.breakdown("byakuya's bankai-transform"))
print()
print("  bankai as a whole has", len([k for r, k in b.structs['bankai'].links if r == 'has-kind']),
      "kept instances; each becoming is different; the frame's transform slot is 'whatever varies here'.")
print("  a change: a transformation OF the before, BY the command, INTO the after - with the becoming kept too.")

section("3. the roles came with their typing: what has filled each slot")
for filler, role in (("sword", "weapon"), ("zangetsu-sealed", "weapon"), ("release", "command"), ("blade-widens-in-light", "command")):
    lik, why = b.fit(filler, role)
    print(f"  can {filler} be the {role}?  {lik:.2f}  ({why})")

section("4. a sword with a button that becomes a spiked gauntlet")
seen = {"weapon": "sword", "command": "button", "transform": "transform", "new-form": "spiked-gauntlet"}
for cov, lik, frame, missing, fits in b.recognize(seen):
    print(f"  shape of '{frame}': {cov:.0%}" + (f" (missing {', '.join(missing)})" if missing else "") + f"   experience: {lik:.2f}")
print("  -> by shape, a shikai (the first four steps of a bankai). the weapon slot likes it: kinds of sword")
print("     have been the weapon. the command slot has never seen a button - possible, unlikely, until it does.")
