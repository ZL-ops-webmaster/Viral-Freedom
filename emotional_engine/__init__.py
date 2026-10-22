"""Emotional learning engine: a learner whose affective state is a set of
weighted values that actively steer its exploration and learning rate."""

from .emotions import EmotionalState, EMOTIONS, appraise
from .engine import EmotionalLearningEngine

__all__ = [
    "EmotionalState",
    "EMOTIONS",
    "appraise",
    "EmotionalLearningEngine",
]
