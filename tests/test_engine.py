import unittest

from emotional_engine import EmotionalLearningEngine, EmotionalState, appraise


class TestEmotionalState(unittest.TestCase):
    def test_nudge_clamps(self):
        s = EmotionalState()
        s.nudge({"joy": 5.0})
        self.assertEqual(s.intensities["joy"], 1.0)
        s.nudge({"joy": -5.0})
        self.assertEqual(s.intensities["joy"], 0.0)

    def test_decay_moves_toward_baseline(self):
        s = EmotionalState(baseline=0.1)
        s.intensities["fear"] = 1.0
        s.decay()
        self.assertLess(s.intensities["fear"], 1.0)
        self.assertGreater(s.intensities["fear"], 0.1)

    def test_valence_signs(self):
        s = EmotionalState()
        s.intensities = {k: 0.0 for k in s.intensities}
        s.intensities["joy"] = 1.0
        self.assertGreater(s.valence(), 0)
        s.intensities["joy"] = 0.0
        s.intensities["sadness"] = 1.0
        self.assertLess(s.valence(), 0)

    def test_unknown_emotion_raises(self):
        with self.assertRaises(KeyError):
            EmotionalState().nudge({"ennui": 0.5})


class TestAppraise(unittest.TestCase):
    def test_positive_reward_lifts_joy(self):
        d = appraise(0.8, 0.0)
        self.assertGreater(d["joy"], 0)
        self.assertLessEqual(d["sadness"], 0)

    def test_negative_reward_lifts_fear(self):
        d = appraise(-0.8, 0.0)
        self.assertGreater(d["fear"], 0)
        self.assertLess(d["joy"], 0)

    def test_prediction_error_drives_surprise(self):
        self.assertGreater(appraise(0.0, 1.0)["surprise"],
                           appraise(0.0, 0.0)["surprise"])


class TestEngineLearns(unittest.TestCase):
    def test_prefers_rewarding_action(self):
        eng = EmotionalLearningEngine(["good", "bad"], seed=1)
        rewards = {"good": 1.0, "bad": -1.0}
        for _ in range(300):
            eng.act_and_learn("s", lambda st, a: rewards[a])
        prefs = eng.preferences("s")
        self.assertGreater(prefs["good"], prefs["bad"])

    def test_fear_reduces_exploration(self):
        eng = EmotionalLearningEngine(["a", "b"], seed=2)
        calm = eng._exploration_rate()
        eng.emotion.intensities["fear"] = 1.0
        scared = eng._exploration_rate()
        self.assertLess(scared, calm)

    def test_arousal_raises_learning_rate(self):
        # Arousal is a normalized weighted average, so a high-arousal emotion
        # (surprise) must raise the rate above a low-arousal one (sadness).
        eng = EmotionalLearningEngine(["a"], seed=3)
        for k in eng.emotion.intensities:
            eng.emotion.intensities[k] = 0.01
        eng.emotion.intensities["sadness"] = 1.0
        low = eng._learning_rate()
        for k in eng.emotion.intensities:
            eng.emotion.intensities[k] = 0.01
        eng.emotion.intensities["surprise"] = 1.0
        high = eng._learning_rate()
        self.assertGreater(high, low)

    def test_empty_actions_rejected(self):
        with self.assertRaises(ValueError):
            EmotionalLearningEngine([])


if __name__ == "__main__":
    unittest.main()
