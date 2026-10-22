"""Demonstrate the volunteer swarm end to end.

Run:
    python3 -m emotional_engine.swarm_demo

You will see:
  1. Volunteers opt in and each process tiny slices; the shared engine learns.
  2. Most volunteers opt out; the rest keep it going.
  3. Everyone opts out -> the swarm is no longer alive and is torn down.
"""

from __future__ import annotations

import random

from .swarm import Application, Coordinator, VolunteerNode

TRUE_REWARD = {"meadow": 0.8, "thicket": 0.0, "ravine": -0.9}


def forager_reward(action, rng):
    return max(-1.0, min(1.0, TRUE_REWARD[action] + rng.gauss(0, 0.1)))


def preferences_line(coord):
    prefs = coord.shared_preferences()
    return "  ".join(f"{k}={v:+.2f}" for k, v in sorted(prefs.items()))


def main() -> None:
    rng = random.Random(42)
    app = Application(
        name="forager",
        state="forest",
        actions=("meadow", "thicket", "ravine"),
        reward=forager_reward,
    )
    coord = Coordinator(app)
    nodes = [VolunteerNode(f"node-{i}", app) for i in range(5)]

    print("== phase 1: five volunteers opt in, each does tiny work units ==")
    for n in nodes:
        n.opt_in()
    for rnd in range(1, 9):
        units = coord.make_units(count=10, steps=4, rng=rng)
        done = coord.run_round(nodes, units)
        print(f"round {rnd:>2}: active={len(coord.active_nodes(nodes))} "
              f"units_done={done:>2}  shared[{preferences_line(coord)}]")

    print("\n== phase 2: three volunteers opt out; two keep it alive ==")
    for n in nodes[:3]:
        n.opt_out()
    for rnd in range(9, 13):
        units = coord.make_units(count=10, steps=4, rng=rng)
        done = coord.run_round(nodes, units)
        print(f"round {rnd:>2}: active={len(coord.active_nodes(nodes))} "
              f"units_done={done:>2}  shared[{preferences_line(coord)}]")

    print(f"\nshared engine still alive? {coord.alive(nodes)}")
    print(f"total steps processed across the swarm: {coord.total_steps}")

    print("\n== phase 3: everyone opts out -> it dies ==")
    for n in nodes:
        n.opt_out()
    units = coord.make_units(count=10, steps=4, rng=rng)
    done = coord.run_round(nodes, units)
    print(f"units processed with zero volunteers: {done}")
    print(f"alive? {coord.alive(nodes)}")
    coord.shutdown()
    print("coordinator.shutdown() called: shared state discarded.")
    print(f"shared preferences after shutdown: [{preferences_line(coord)}]")
    print(f"alive? {coord.alive(nodes)}")


if __name__ == "__main__":
    main()
