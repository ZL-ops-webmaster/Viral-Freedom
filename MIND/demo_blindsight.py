"""Blindsight: recognition survives when the percept is lesioned (design.md §2a, §28a).

A sense (recognition) and a percept (the conscious report) are different stages.
Damage the report and the sense still fires -- the patient says they see
nothing, yet reaches to the right colored ball. Here: the brain keeps selecting
the correct ball with the seeing cortex lesioned, but can no longer say what it
sees.

Run:
    python3 demo_blindsight.py
"""

from __future__ import annotations

import numpy as np
import brain as B
from brain import Brain

# A tiny colour world (configure points the mind at a different substrate).
B.configure(
    features=["red", "green", "blue", "size"],
    descriptor_words={
        "red": ("not-red", "red"), "green": ("not-green", "green"),
        "blue": ("not-blue", "blue"), "size": ("small", "big"),
    },
)

BALLS = {
    "red-ball":   np.array([9.0, 1.0, 1.0, 5.0]),
    "green-ball": np.array([1.0, 9.0, 1.0, 5.0]),
    "blue-ball":  np.array([1.0, 1.0, 9.0, 5.0]),
}


def main() -> None:
    b = Brain(log=lambda *a: None)
    # Teach the three ball senses (a few each so the word binds).
    rng = np.random.default_rng(0)
    for name, proto in BALLS.items():
        for _ in range(3):
            b.learn(proto + rng.normal(0, 0.2, 4), name.split("-")[0])  # word = colour

    target = BALLS["red-ball"] + rng.normal(0, 0.2, 4)

    print("== seeing INTACT ==")
    sense = b.nearest(target)
    print(f"   recognition selects: {sense.label}   (the sense -> could grab it)")
    print(f"   percept it reports : {b.describe(target)!r}   (the conscious 'seeing')")

    print("\n== seeing LESIONED (blindsight) ==")
    b.seeing.lesion()
    sense = b.nearest(target)
    print(f"   recognition selects: {sense.label}   (STILL correct -> still grabs the right ball)")
    print(f"   percept it reports : {b.describe(target)!r}   (gone -- 'I can't see anything')")

    picked = b.nearest(target).label                 # the bound word is the colour: 'red'
    print(f"\n   discrimination among 3 balls, lesioned: picked {picked!r} "
          f"-> {'correct' if picked == 'red' else 'WRONG'}")
    print("   the sense acts; the report is dark. recognition and seeing are different stages.")


if __name__ == "__main__":
    main()
