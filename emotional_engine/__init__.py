"""Emotional learning engine: a learner whose affective state is a set of
weighted values that actively steer its exploration and learning rate, plus a
volunteer swarm and an opt-in, read-only web sense."""

from .emotions import EmotionalState, EMOTIONS, appraise
from .engine import EmotionalLearningEngine
from .swarm import Application, Coordinator, VolunteerNode, WorkUnit
from .websense import WebConsent, WebObservation, WebSense, appraise_text

__all__ = [
    "EmotionalState",
    "EMOTIONS",
    "appraise",
    "EmotionalLearningEngine",
    "Application",
    "Coordinator",
    "VolunteerNode",
    "WorkUnit",
    "WebConsent",
    "WebObservation",
    "WebSense",
    "appraise_text",
]
