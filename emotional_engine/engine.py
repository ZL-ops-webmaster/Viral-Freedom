"""An emotional learning engine.

The engine learns which action to take in a given situation by trial and
error. What makes it *emotional* is that its affective state is not cosmetic:
it is a set of weighted values that steer the learning itself.

  - valence (pleasant/unpleasant) biases which actions look appealing;
  - arousal (calm/activated) scales the learning rate, so emotionally
    charged moments are learned from more sharply;
  - fear specifically suppresses exploration, making the agent cautious;
  - joy / anticipation widen exploration, making it curious.

Each experience updates both the value estimates and the emotional state,
and the emotional state in turn shapes the next decision. The loop is the
point: emotion and learning feed each other.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Hashable, List, Sequence, Tuple

from .emotions import EmotionalState, appraise


class EmotionalLearningEngine:
    def __init__(
        self,
        actions: Sequence[Hashable],
        *,
        base_learning_rate: float = 0.15,
        discount: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if not actions:
            raise ValueError("need at least one action")
        self.actions: List[Hashable] = list(actions)
        self.base_learning_rate = base_learning_rate
        self.discount = discount
        self.rng = random.Random(seed)
        self.emotion = EmotionalState()

        # value[state][action] -> running estimate of expected reward.
        self.value: Dict[Hashable, Dict[Hashable, float]] = defaultdict(
            lambda: {a: 0.0 for a in self.actions}
        )
        self.steps = 0

    # -- policy ------------------------------------------------------------

    def _exploration_rate(self) -> float:
        """How often to try a non-greedy action, shaped by emotion.

        Fear narrows exploration (play it safe); joy and anticipation widen
        it (seek novelty). Bounded so the agent never fully stops learning
        and never acts purely at random.
        """
        curiosity = self.emotion.intensities["joy"] + \
            self.emotion.intensities["anticipation"]
        caution = self.emotion.intensities["fear"]
        raw = 0.15 + 0.35 * curiosity - 0.30 * caution
        return max(0.02, min(0.8, raw))

    def _learning_rate(self) -> float:
        """Arousal sharpens learning: charged moments update estimates more."""
        return self.base_learning_rate * (1.0 + self.emotion.arousal())

    def decide(self, state: Hashable) -> Hashable:
        """Pick an action for `state` under the current emotional state."""
        eps = self._exploration_rate()
        if self.rng.random() < eps:
            return self.rng.choice(self.actions)

        q = self.value[state]
        # Valence tilts the greedy choice: a positive mood adds a small
        # optimistic bonus that fades as estimates become confident.
        tilt = 0.1 * self.emotion.valence()
        best, best_score = None, -math.inf
        for a in self.actions:
            score = q[a] + tilt
            if score > best_score:
                best, best_score = a, score
        return best

    # -- learning ----------------------------------------------------------

    def learn(self, state: Hashable, action: Hashable, reward: float) -> None:
        """Update value estimate and emotional state from one outcome."""
        q = self.value[state]
        expected = q[action]
        prediction_error = reward - expected

        lr = self._learning_rate()
        q[action] = expected + lr * prediction_error

        # Feed the outcome back into emotion, then let emotions relax a step.
        self.emotion.nudge(appraise(reward, abs(prediction_error)))
        self.emotion.decay()
        self.steps += 1

    def act_and_learn(
        self, state: Hashable, reward_fn
    ) -> Tuple[Hashable, float]:
        """Convenience: decide, observe reward via reward_fn(state, action),
        then learn. Returns the (action, reward) taken."""
        action = self.decide(state)
        reward = reward_fn(state, action)
        self.learn(state, action, reward)
        return action, reward

    # -- inspection --------------------------------------------------------

    def preferences(self, state: Hashable) -> Dict[Hashable, float]:
        return dict(self.value[state])

    def mood(self) -> str:
        return str(self.emotion)

    # -- serialization (for the volunteer swarm) ---------------------------

    def export_state(self) -> dict:
        """A plain, serializable snapshot of what this node has learned."""
        return {
            "value": {s: dict(a) for s, a in self.value.items()},
            "emotion": self.emotion.snapshot(),
            "steps": self.steps,
        }

    def import_state(self, state: dict) -> None:
        """Load a shared snapshot in, replacing local estimates."""
        for s, a in state.get("value", {}).items():
            self.value[s] = {act: a.get(act, 0.0) for act in self.actions}
        emo = state.get("emotion")
        if emo:
            for k, v in emo.items():
                if k in self.emotion.intensities:
                    self.emotion.intensities[k] = v
        self.steps = state.get("steps", self.steps)
