"""Learned grammar in front of the brain.

Run:
    python3 demo_grammar.py

Shows, end to end:
  1. It LEARNS a sentence shape from a couple of examples (no hardcoded rule).
  2. It PARSES a sentence it has never seen, of that shape.
  3. It SPEAKS the shape with unknown words -- before understanding them.
  4. Meaning arrives LATER: the brain's intuit() places an unknown word from
     the company it kept (the Japanese analogy, start to finish).
  5. It PRODUCES a sentence from a meaning and reads its own words back.
"""

from __future__ import annotations

from brain import Brain
from grammar import Grammar


def main() -> None:
    log = []
    brain = Brain(log=lambda *a: None)      # quiet
    g = Grammar()

    # The brain knows these frames (what a sentence can MEAN).
    brain.frame("give", {"giver": "?", "thing": "?", "receiver": "?"},
                "someone hands a thing to someone else")
    brain.frame("cut", {"cutter": "?", "thing": "?", "tool": "?"},
                "someone divides a thing using a tool")

    print("== 1. EXPOSURE: hear a shape a few times ==")
    for s, f in [
        ("ichigo gave rukia a sword", {"giver": "ichigo", "receiver": "rukia", "thing": "sword"}),
        ("the knight gave the king a crown", {"giver": "knight", "receiver": "king", "thing": "crown"}),
        ("the cook cut the bread with a knife", {"cutter": "cook", "thing": "bread", "tool": "knife"}),
        ("the knight cut the rope with his sword", {"cutter": "knight", "thing": "rope", "tool": "sword"}),
    ]:
        g.teach(s, f["frame"] if "frame" in f else ("give" if "giver" in f else "cut"), f)
        print(f"   heard: {s!r}")
    print("   learned patterns:")
    for k in g.known():
        print(f"     {k}")

    print("\n== 2. PARSE a sentence never seen, of a known shape ==")
    r = g.parse("renji gave byakuya a blade")
    print(f"   'renji gave byakuya a blade' -> {r['frame']}  {r['fillers']}")
    brain.instantiate("renji's give #1", r["frame"], r["fillers"])

    print("\n== 3. SPEAK before understanding: all-nonsense fillers still bind ==")
    r2 = g.parse("zylo cut vorn with a florb")
    print(f"   'zylo cut vorn with a florb' -> {r2['frame']}  {r2['fillers']}")
    print("   (the shape is spoken; 'florb' means nothing yet)")

    print("\n== 4. MEANING LATER: parse surfaces what it doesn't understand ==")
    # Parsing never needs to know what a filler means. The words it binds that
    # the brain has no concept for are exactly the ones to understand later --
    # the adapter hands them off; resolving them is the brain's intuit(), which
    # places a word from the company it keeps over a few uses. (Proven for real
    # in talk.py's transcript: a 'zanpakuto', never defined, is intuited to be a
    # kind of sword from 'tool of a cut' + 'has a handle' + 'like a whip'.)
    known = set(brain.structs)
    unknown = sorted({w for w in r2["fillers"].values() if w not in known})
    print(f"   parsed fillers: {r2['fillers']}")
    print(f"   not understood yet (-> brain.intuit over more uses): {unknown}")

    print("\n== 5. PRODUCE a meaning, then read it back ==")
    said = g.produce("give", {"giver": "renji", "receiver": "byakuya", "thing": "blade"})
    print(f"   produce(give, ...) -> {said!r}")
    back = g.parse(said)
    print(f"   parse(that)        -> {back['frame']}  {back['fillers']}")


if __name__ == "__main__":
    main()
