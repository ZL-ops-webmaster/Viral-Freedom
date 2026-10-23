import unittest
from grammar import Grammar


class TestLearnAndParse(unittest.TestCase):
    def setUp(self):
        self.g = Grammar()
        # Hear a shape a few times (exposure), never hardcoded.
        self.g.teach("ichigo gave rukia a sword", "give",
                     {"giver": "ichigo", "receiver": "rukia", "thing": "sword"})
        self.g.teach("the knight gave the king a crown", "give",
                     {"giver": "knight", "receiver": "king", "thing": "crown"})

    def test_induces_one_pattern(self):
        # Two examples of the same shape merge into a single learned pattern.
        gives = [p for p in self.g.patterns if p.frame == "give"]
        self.assertEqual(len(gives), 1)
        self.assertEqual(gives[0].examples, 2)

    def test_parses_a_novel_sentence(self):
        r = self.g.parse("renji gave byakuya a blade")
        self.assertEqual(r["frame"], "give")
        self.assertEqual(r["fillers"],
                         {"giver": "renji", "receiver": "byakuya", "thing": "blade"})

    def test_speak_before_understand(self):
        # Unknown nonsense words still bind to slots: the shape is spoken
        # before any meaning is known.
        r = self.g.parse("zylo gave vorn a florb")
        self.assertEqual(r["fillers"],
                         {"giver": "zylo", "receiver": "vorn", "thing": "florb"})

    def test_rejects_wrong_shape(self):
        self.assertIsNone(self.g.parse("ichigo ran"))


class TestPrepositionsFallOutFree(unittest.TestCase):
    def test_with_phrase_becomes_an_anchor(self):
        g = Grammar()
        g.teach("the cook cut the bread with a knife", "cut",
                {"cutter": "cook", "thing": "bread", "tool": "knife"})
        # A brand-new sentence of the same shape, different words.
        r = g.parse("the knight cut the rope with his sword")
        self.assertEqual(r["frame"], "cut")
        self.assertEqual(r["fillers"],
                         {"cutter": "knight", "thing": "rope", "tool": "sword"})


class TestProduce(unittest.TestCase):
    def test_says_a_meaning(self):
        g = Grammar()
        g.teach("ichigo gave rukia a sword", "give",
                {"giver": "ichigo", "receiver": "rukia", "thing": "sword"})
        said = g.produce("give",
                         {"giver": "renji", "receiver": "byakuya", "thing": "blade"})
        self.assertIn("gave", said)
        self.assertIn("renji", said)
        self.assertIn("blade", said)
        # Round-trip: what it says, it can read back.
        self.assertEqual(g.parse(said)["frame"], "give")


class TestTwoShapesCoexist(unittest.TestCase):
    def test_picks_the_right_frame(self):
        g = Grammar()
        g.teach("ichigo gave rukia a sword", "give",
                {"giver": "ichigo", "receiver": "rukia", "thing": "sword"})
        g.teach("the cook cut the bread with a knife", "cut",
                {"cutter": "cook", "thing": "bread", "tool": "knife"})
        self.assertEqual(g.parse("renji gave byakuya a blade")["frame"], "give")
        self.assertEqual(g.parse("the knight cut the rope with his sword")["frame"], "cut")


if __name__ == "__main__":
    unittest.main()


class TestAutonomousInduction(unittest.TestCase):
    """observe(): learn shapes from raw sentences, no roles marked."""

    def setUp(self):
        self.g = Grammar()
        self.g.observe([
            "ichigo gave rukia a sword",
            "the knight gave the king a crown",
            "renji gave byakuya a blade",
        ])

    def test_induces_verb_framed_pattern(self):
        gives = [p for p in self.g.patterns if p.frame == "gave"]
        self.assertEqual(len(gives), 1)
        # subject, receiver, thing vary -> 3 slots; "gave" is the one anchor.
        self.assertEqual(gives[0].slots(), ["s0", "s1", "s2"])
        lits = [t.value for t in gives[0].toks if t.kind == "lit"]
        self.assertEqual(lits, ["gave"])

    def test_parses_novel_unlabelled(self):
        r = self.g.parse("zaraki gave yachiru a candy")
        self.assertEqual(r["frame"], "gave")
        self.assertEqual(r["fillers"], {"s0": "zaraki", "s1": "yachiru", "s2": "candy"})

    def test_preposition_becomes_anchor_unsupervised(self):
        g = Grammar()
        g.observe([
            "the cook cut the bread with a knife",
            "the knight cut the rope with a sword",
            "ichigo cut the hollow with a zanpakuto",
        ])
        cuts = [p for p in g.patterns if p.frame == "cut"]
        self.assertEqual(len(cuts), 1)
        lits = [t.value for t in cuts[0].toks if t.kind == "lit"]
        self.assertEqual(lits, ["cut", "with"])           # verb AND preposition, unaided

    def test_meaning_attached_later(self):
        # The shape was learned unlabelled; narration names the slots after.
        self.g.name_slots("gave", {"s0": "giver", "s1": "receiver", "s2": "thing"})
        r = self.g.parse("zaraki gave yachiru a candy")
        self.assertEqual(r["fillers"],
                         {"giver": "zaraki", "receiver": "yachiru", "thing": "candy"})

    def test_single_example_is_not_a_shape(self):
        g = Grammar()
        made = g.observe(["ichigo gave rukia a sword"])   # one example only
        self.assertEqual(made, [])
