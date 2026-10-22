"""Affective state as weighted values.

The model is a small vector of named emotions, each an intensity in [0, 1].
Emotions decay toward a resting baseline over time, can be nudged by appraised
events, and expose derived quantities (valence, arousal) that the learning
engine uses to modulate how it explores and how fast it updates.

Design goals:
- Pure stdlib, deterministic given a seed, easy to inspect and unit test.
- Emotions are *weights* on behaviour, not decoration: valence and arousal
  feed directly into the engine's learning rate and exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

# Plutchik-style primaries. Each is paired conceptually with an opposite,
# but we track them independently and let blends emerge.
EMOTIONS = (
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "anger",
    "anticipation",
)

# How each emotion contributes to overall valence (pleasant vs unpleasant)
# and arousal (calm vs activated). Values are hand-set priors in [-1, 1].
_VALENCE = {
    "joy": 1.0,
    "trust": 0.6,
    "anticipation": 0.3,
    "surprise": 0.0,
    "fear": -0.7,
    "sadness": -0.8,
    "anger": -0.6,
}
_AROUSAL = {
    "joy": 0.5,
    "trust": 0.1,
    "anticipation": 0.6,
    "surprise": 0.9,
    "fear": 0.9,
    "sadness": 0.2,
    "anger": 0.8,
}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class EmotionalState:
    """A mutable bundle of weighted emotion intensities."""

    # Resting level each emotion drifts back toward.
    baseline: float = 0.1
    # Fraction of the gap to baseline closed per decay() call.
    decay_rate: float = 0.15
    intensities: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in EMOTIONS:
            self.intensities.setdefault(name, self.baseline)

    # -- updates -----------------------------------------------------------

    def nudge(self, deltas: Mapping[str, float]) -> None:
        """Add signed changes to named emotions, clamped to [0, 1]."""
        for name, d in deltas.items():
            if name not in self.intensities:
                raise KeyError(f"unknown emotion: {name!r}")
            self.intensities[name] = _clamp(self.intensities[name] + d)

    def decay(self) -> None:
        """Relax every emotion a step toward baseline."""
        for name, v in self.intensities.items():
            self.intensities[name] = v + (self.baseline - v) * self.decay_rate

    # -- derived quantities ------------------------------------------------

    def valence(self) -> float:
        """Net pleasantness in roughly [-1, 1]."""
        total = sum(self.intensities.values()) or 1.0
        s = sum(self.intensities[n] * _VALENCE[n] for n in EMOTIONS)
        return _clamp(s / total, -1.0, 1.0)

    def arousal(self) -> float:
        """Overall activation in [0, 1]."""
        total = sum(self.intensities.values()) or 1.0
        s = sum(self.intensities[n] * _AROUSAL[n] for n in EMOTIONS)
        return _clamp(s / total, 0.0, 1.0)

    def dominant(self) -> str:
        """Name of the strongest current emotion."""
        return max(self.intensities, key=self.intensities.get)

    def snapshot(self) -> Dict[str, float]:
        return dict(self.intensities)

    def __str__(self) -> str:
        parts = ", ".join(f"{n}={v:.2f}" for n, v in self.intensities.items())
        return (
            f"<{self.dominant()} | val={self.valence():+.2f} "
            f"aro={self.arousal():.2f} | {parts}>"
        )


def appraise(reward: float, prediction_error: float) -> Dict[str, float]:
    """Turn an outcome into emotional nudges.

    reward: how good the outcome was, roughly [-1, 1].
    prediction_error: |expected - actual|, >= 0; drives surprise.

    This is the appraisal layer: it decides which emotions an event feeds.
    """
    deltas: Dict[str, float] = {}
    if reward >= 0:
        deltas["joy"] = 0.35 * reward
        deltas["trust"] = 0.20 * reward
        deltas["anticipation"] = 0.10 * reward
        deltas["sadness"] = -0.15 * reward
        deltas["fear"] = -0.10 * reward
    else:
        mag = -reward
        deltas["sadness"] = 0.30 * mag
        deltas["fear"] = 0.25 * mag
        deltas["anger"] = 0.20 * mag
        deltas["joy"] = -0.20 * mag
        deltas["trust"] = -0.10 * mag

    deltas["surprise"] = _clamp(0.6 * prediction_error, 0.0, 0.6)
    return deltas
