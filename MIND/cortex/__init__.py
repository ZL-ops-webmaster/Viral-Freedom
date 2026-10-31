"""Cortices — the brain, split into regions that pass signals.

B.R.A.I.N. began as one object (brain.py) holding every kind of state at once:
the perceptual concept tree, the structural/language layer, feeling, wanting,
memory. That is one file you have to read all of to understand any of. A real
brain is regions that each do one thing and COMMUNICATE: the visual cortex
hands shapes to recognition, recognition hands a surprise to the limbic
system, and so on.

This package pulls those regions out one at a time, each owning its own state
and exposing a small interface the others talk to. The split is incremental and
behaviour-preserving: every extraction keeps the demos green. The Brain becomes
the coordinator that wires the cortices together.

Planned regions (carved as each proves clean):
  - feeling    — valence that directs attention (surprise, satisfaction,
                 unease, curiosity).                           [EXTRACTED]
  - perception — the concept tree: senses, nearest, split/compose, memory.
  - language   — the structural layer: structs, frames, episodes, intuition.
  - motivation — wanting: desires derived from feeling.
  - thinking   — prediction, imagination, wonder, the idle loop.
"""

from .feeling import Feeling

__all__ = ["Feeling"]
