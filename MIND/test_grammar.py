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
