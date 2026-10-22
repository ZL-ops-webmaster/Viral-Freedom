"""B.R.A.I.N. demo 14: ROLES - the sum and the parts, at once.

  "When you say 'Bankai' I think of the parts: a transformation of a weapon, with a specific command,
   into a new form. And the sum: it is Bankai. If I saw a sword with a button that changed it into a
   gauntlet with spikes - I can see that, it IS a Bankai. And it's its own thing as well. Its general
   term is a transforming weapon."
  - frame():       a structure WITH its relations (roles), plus the story
  - recognize():   role-FILLING, not overlap - "it is a bankai"
  - instantiate(): it's a bankai AND its own thing
  - explain():     break it down for someone who has never heard it
Run:  ..\\WOLF\\.venv-pose\\Scripts\\python.exe demo_bankai.py
"""
from brain import Brain

def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
b = Brain(log=print)

section("1. the general term, then the specific structure (with its story)")
b.frame("transforming-weapon", {"weapon": None, "transformation": None})
b.frame("bankai", {"weapon": None, "command": None, "transformation": None, "new-form": None},
        story="a transformation of a weapon, triggered by a specific command, into a new form (Bleach)")

section("2. 'I saw a sword with a button that turned it into a gauntlet with spikes'")
seen = {"weapon": "sword", "command": "button", "transformation": "changes-form", "new-form": "spiked-gauntlet"}
for cov, lik, frame, missing, fits in b.recognize(seen):
    print(f"  the SHAPE of '{frame}': {cov:.0%} of its roles filled" + (f" (missing {', '.join(missing)})" if missing else "")
          + f"   experience says: {lik:.2f}")
print("  -> by shape it is a bankai (generally, a transforming weapon). by experience: it has never seen")
print("     anything fill these roles, so it can't yet say how likely. accumulation fixes that (section 6).")

section("3. ...and it's its own thing as well")
b.instantiate("spike-sword", "bankai", seen)
print("  spike-sword is:", [o for r, o in b.structs["spike-sword"].links if r == "is-a"], "->",
      [o for r, o in b.structs["bankai"].links if r == "is-a"])

section("4. explaining it to someone who has never heard it")
print(b.breakdown("bankai"))
print()
print(b.breakdown("spike-sword"))

section("5. telling it to someone: the shortest thing THEY can decompress")
everyday = {"sword", "button", "spiked-gauntlet", "changes-form"}
bleach_fan = everyday | {"bankai", "transforming-weapon"}
weapons_nerd = everyday | {"transforming-weapon"}
print(f"  to someone who doesn't watch anime:  'it's {b.tell('spike-sword', everyday)}. isn't that cool?'")
print(f"  to a weapons nerd:                   'it's {b.tell('spike-sword', weapons_nerd)}'")
print(f"  to someone who watched Bleach:       'check it out - {b.tell('spike-sword', bleach_fan)} in real life'")

section("6. slot fit is GRADED, from accumulated simple connections - never a type check")
b.equate("transformation", "change")                                   # transform = change
b.instantiate("detonator", "bankai", {"weapon": "bomb", "command": "button", "transformation": "explodes", "new-form": "crater"})
b.instantiate("thrown-button", "transforming-weapon", {"weapon": "button", "transformation": "flies-at-head"})
b.frame("switch", {"thing": "button", "change": "flip"})              # a button CAN transform: down -> up
print()
for filler, role in (("button", "command"), ("button", "weapon"), ("button", "transformation"), ("sword", "command")):
    lik, why = b.fit(filler, role)
    print(f"  can a {filler} be the {role}?  {lik:.2f}  ({why})")
print()
wrong = {"weapon": "button", "command": "sword", "transformation": "spiked-gauntlet", "new-form": "changes-form"}
right = b.recognize(seen)[0]; scrambled = b.recognize(wrong)[0]
print(f"  the sword with a button, roles as seen:  bankai, likelihood {right[1]:.2f}")
print(f"  same four parts, roles scrambled:        bankai, likelihood {scrambled[1]:.2f}  <- possible, unlikely")
print("\n  'I know a button CAN be a weapon. But it's more than likely a trigger.'")
print("  simple connections, accumulated, make major compressions.")
