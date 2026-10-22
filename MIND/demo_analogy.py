"""B.R.A.I.N. demo 13: ANALOGY - a handle is a library.

  "Senbonzakura: one word, a specific structure - sword, petals, projectile, safe zone, reach, many,
   controlled. With that one word I could describe a thousand things. A gauntlet that shoots a fan of
   darts: 'oh, like Byakuya's bankai' - 'yes, exactly like Senbonzakura'. But I could also use it to
   describe just the safe zone, and make a robot only operate in an area the user permits."
  - like():   whose structure overlaps mine, wherever they sit
  - borrow(): take one addressable part across, however unrelated the target
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_analogy.py
"""
from brain import Brain

def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
b = Brain(log=print)

section("1. one word, a whole structure")
b.define("safe-zone", ["region", "no-entry", "owner-override"])
b.define("senbonzakura", ["sword", "petals", "projectile", "safe-zone", "reach", "many", "controlled"])

section("2. a gauntlet that shoots a fan of darts on a hand gesture")
b.define("gauntlet", ["hand", "gesture", "darts", "projectile", "fan", "many", "controlled"])
for overlap, other, shared, could in b.like("gauntlet"):
    print(f"  'oh, like {other}'  (overlap {overlap:.2f})   shared: {', '.join(shared)}")
    print(f"      what it could borrow: {', '.join(could)}")
b.borrow("gauntlet", "senbonzakura", "safe-zone")
b.borrow("gauntlet", "senbonzakura", "reach")
print("  'yes, exactly like senbonzakura' ->", ", ".join(sorted(b.structure_of("gauntlet"))))

section("3. ...or just one part of it, on something unrelated")
b.define("robot-arm", ["motor", "joint", "grip", "reach"])
print("  like anything?", [o for _, o, _, _ in b.like("robot-arm")])
b.borrow("robot-arm", "senbonzakura", "safe-zone")
print("  robot-arm is now:", ", ".join(sorted(b.structure_of("robot-arm"))))
print("  i.e. it only operates in a region, no-entry by default, unless the owner overrides.")
print("\n  same handle, two transfers: the gauntlet took the structure; the robot took one part.")
