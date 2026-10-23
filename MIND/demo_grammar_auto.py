"""Autonomous grammar induction: learn shapes from raw sentences, no labels.

Run:
    python3 demo_grammar_auto.py

Nobody marks which word is the giver. The adapter hears a few sentences, strips
the function words, groups by the verb, and reads slots off the columns that
vary. Meaning is attached afterwards -- the shape is learned first.
"""

from __future__ import annotations

from grammar import Grammar


def main() -> None:
    g = Grammar()

    print("== hears raw sentences (no roles given) ==")
    corpus = [
        "ichigo gave rukia a sword",
        "the knight gave the king a crown",
        "renji gave byakuya a blade",
        "the cook cut the bread with a knife",
        "the knight cut the rope with a sword",
        "ichigo cut the hollow with a zanpakuto",
    ]
    for s in corpus:
        print(f"   {s!r}")
    g.observe(corpus)

    print("\n== shapes it induced on its own ==")
    for k in g.known():
        print(f"   {k}")

    print("\n== parses a sentence it never heard, of an induced shape ==")
    for s in ("zaraki gave yachiru a candy", "the butcher cut the meat with a cleaver"):
        r = g.parse(s)
        print(f"   {s!r} -> {r['frame']}  {r['fillers']}")

    print("\n== meaning attached later (narration names the slots) ==")
    g.name_slots("gave", {"s0": "giver", "s1": "receiver", "s2": "thing"})
    r = g.parse("zaraki gave yachiru a candy")
    print(f"   after naming: {r['fillers']}")


if __name__ == "__main__":
    main()
