"""A small runnable demo of the emotional learning engine.

Scenario: a forager in a world of three patches. One patch is reliably good,
one is reliably bad (a predator), one is neutral. The agent has never seen
them. Watch its value estimates converge *and* its emotional state move as it
learns -- fear after a bad patch, joy and trust as the good patch pays off.

Run:
    python -m emotional_engine.demo
"""

from __future__ import annotations

import random

from .engine import EmotionalLearningEngine

PATCHES = ("meadow", "thicket", "ravine")
TRUE_REWARD = {"meadow": 0.8, "thicket": 0.0, "ravine": -0.9}


def reward_fn(rng: random.Random):
    def f(state, action):
        # Noisy reward around the patch's true value.
        return max(-1.0, min(1.0, TRUE_REWARD[action] + rng.gauss(0, 0.1)))
    return f


def main() -> None:
    rng = random.Random(7)
    engine = EmotionalLearningEngine(PATCHES, seed=7)
    rf = reward_fn(rng)

    print("step  chose     reward   dominant     valence  arousal")
    print("-" * 60)
    for step in range(1, 61):
        action, reward = engine.act_and_learn("forest", rf)
        if step <= 10 or step % 10 == 0:
            e = engine.emotion
            print(
                f"{step:>4}  {action:<8}  {reward:+.2f}    "
                f"{e.dominant():<10}  {e.valence():+.2f}    {e.arousal():.2f}"
            )

    print("\nlearned value of each patch (higher = more preferred):")
    for patch, v in sorted(
        engine.preferences("forest").items(), key=lambda kv: -kv[1]
    ):
        bar = "#" * int(round((v + 1) * 15))
        print(f"  {patch:<8} {v:+.2f}  {bar}")

    print(f"\nfinal mood: {engine.mood()}")
    print(f"total steps: {engine.steps}")


if __name__ == "__main__":
    main()
