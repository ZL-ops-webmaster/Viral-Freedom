import random
import unittest

from emotional_engine.swarm import Application, Coordinator, VolunteerNode


def _app():
    rewards = {"good": 1.0, "bad": -1.0}
    return Application(
        name="t",
        state="s",
        actions=("good", "bad"),
        reward=lambda a, rng: rewards[a],
    )


class TestSwarm(unittest.TestCase):
    def setUp(self):
        self.app = _app()
        self.coord = Coordinator(self.app)
        self.nodes = [VolunteerNode(f"n{i}", self.app) for i in range(4)]
        self.rng = random.Random(0)

    def test_dies_with_no_volunteers(self):
        # Nobody opted in: not alive, and a round does no work.
        self.assertFalse(self.coord.alive(self.nodes))
        units = self.coord.make_units(5, 4, self.rng)
        self.assertEqual(self.coord.run_round(self.nodes, units), 0)

    def test_alive_with_one_volunteer(self):
        self.nodes[0].opt_in()
        self.assertTrue(self.coord.alive(self.nodes))

    def test_swarm_learns(self):
        for n in self.nodes:
            n.opt_in()
        for _ in range(10):
            units = self.coord.make_units(8, 5, self.rng)
            self.coord.run_round(self.nodes, units)
        prefs = self.coord.shared_preferences()
        self.assertGreater(prefs["good"], prefs["bad"])

    def test_opted_out_node_abandons_unit(self):
        self.nodes[0].opt_in()
        unit = self.coord.make_units(1, 4, self.rng)[0]
        self.nodes[0].opt_out()
        self.assertIsNone(self.nodes[0].process(unit, self.coord.shared))

    def test_shutdown_discards_state(self):
        for n in self.nodes:
            n.opt_in()
        for _ in range(5):
            self.coord.run_round(self.nodes, self.coord.make_units(8, 5, self.rng))
        for n in self.nodes:
            n.opt_out()
        self.coord.shutdown()
        self.assertFalse(self.coord.alive(self.nodes))
        prefs = self.coord.shared_preferences()
        self.assertTrue(all(abs(v) < 1e-9 for v in prefs.values()))

    def test_work_units_are_plain_data(self):
        unit = self.coord.make_units(1, 4, self.rng)[0]
        self.assertIsInstance(unit.seed, int)
        self.assertIsInstance(unit.steps, int)


if __name__ == "__main__":
    unittest.main()
