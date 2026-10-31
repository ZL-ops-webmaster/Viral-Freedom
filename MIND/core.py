"""core — the shared substrate of B.R.A.I.N.

The primitives every cortex operates over: the perceptual feature schema (FEATURES,
DESCRIPTOR_WORDS) and its live switch (configure), the tuning constants, the descriptor-naming
helper (dev_word), and the three node types (Concept, Want, Struct). Kept in one module with no
dependency on any cortex, so brain.py and every cortex/ module can import it without a cycle.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import math
import numpy as np


# ---- perceptual substrate -------------------------------------------------------------
FEATURES = ["size", "bulk", "leg_count", "snout_len", "paw_width", "diet", "tail_len", "fur",
            "claws_retract", "whiskers"]
DESCRIPTOR_WORDS = {   # how a deviation along a feature gets *named* as a descriptor concept (neg, pos)
    "size": ("small", "big"), "bulk": ("slender", "broad"), "snout_len": ("short-snout", "long-snout"),
    "paw_width": ("narrow-paws", "wide-paws"), "diet": ("plant-leaning-diet", "meat-leaning-diet"),
    "tail_len": ("short-tail", "long-tail"), "fur": ("thin-fur", "thick-fur"),
    "claws_retract": ("fixed-claws", "retractable-claws"), "whiskers": ("no-whiskers", "whiskers"),
}
def configure(features: list, descriptor_words: dict):
    """Point the mind at a different perceptual substrate (a different world). The mechanism is
    the same; only what it can sense changes. Call before building a Brain."""
    # mutate in place (do NOT rebind) so every module that imported these names keeps the
    # same live objects when the world is switched.
    FEATURES[:] = list(features)
    DESCRIPTOR_WORDS.clear(); DESCRIPTOR_WORDS.update(descriptor_words)


K_TOL = 2.5          # a feature is a descriptor when |x - proto| > K_TOL * spread
SPREAD_FLOOR = 0.6   # a one-example sense still has a tolerance
NODE_COST = 3        # fixed cost of minting a concept node (tokens)
MIN_RECUR = 2        # a kind of bundle must recur before it can earn a node
OVERLAP = 0.5        # Jaccard overlap for two bundles to count as the same recurring kind


def dev_word(dim: str, value: float, ref: float) -> str:
    """Name the descriptor for `value` relative to `ref` along `dim`."""
    if dim == "leg_count":
        return f"{int(round(value))}-legged"
    neg, pos = DESCRIPTOR_WORDS[dim]
    return pos if value > ref else neg


# ---- concept node = a sense --------------------------------------------------------------
@dataclass
class Concept:
    cid: int
    proto: np.ndarray                       # centre of the sense
    m2: np.ndarray                          # running sum of squares (Welford) for spread
    n: int = 1
    parent: Optional["Concept"] = None
    children: list = field(default_factory=list)
    defined_by: tuple = ()                  # core descriptors relative to parent, stored ONCE
    instances: list = field(default_factory=list)   # (x, bundle, word) explained as me ∘ bundle
    word_counts: Counter = field(default_factory=Counter)
    prior: Optional[np.ndarray] = None      # inherited variability: a one-example concept varies like its parent
    born_from: int = 1                      # how many observations it took to EXIST (accumulation metric)
    # CONSOLIDATED feeling: folded in from raw memories as they are forgotten (sleep), so you still
    # like cookies after you've forgotten every specific one. Live feeling is re-read from raw memories.
    v_sum: float = 0.0
    v_n: int = 0
    v_watched: int = 0
    # THE WEB: the tree gives one definitional parent (what makes descriptions short); links give
    # everything else - cross-cutting memberships ("is-a" vertebrate, pet), parts, named-after /
    # resembles ("assoc"). A shark connects to its kinds, its teeth, and the whale shark, which also
    # connects to whales despite the differences.
    links: list = field(default_factory=list)   # (relation, Concept)
    called: set = field(default_factory=set)    # every word this has ever been called (history of belonging)

    @property
    def spread(self) -> np.ndarray:
        if self.n > 1:
            std = np.sqrt(self.m2 / (self.n - 1))
        else:
            std = self.prior if self.prior is not None else np.zeros_like(self.proto)
        return np.maximum(std, SPREAD_FLOOR)

    @property
    def label(self) -> str:                 # the handle, once a word has bound by recurrence
        if self.word_counts:
            w, c = self.word_counts.most_common(1)[0]
            if c >= MIN_RECUR:
                return w
        return f"concept#{self.cid}"

    def absorb(self, x: np.ndarray):        # one-example update of the sense (Welford)
        self.n += 1
        d = x - self.proto
        self.proto = self.proto + d / self.n
        self.m2 = self.m2 + d * (x - self.proto)

    def deviations(self, x: np.ndarray) -> tuple:
        out = []
        for i, dim in enumerate(FEATURES):
            if abs(x[i] - self.proto[i]) > K_TOL * self.spread[i]:
                out.append((dim, 1 if x[i] > self.proto[i] else -1))
        return tuple(out)

    def distance(self, x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(((x - self.proto) / self.spread) ** 2)))

    def relation(self) -> str:              # "parent + core descriptors", read off real prototypes
        if not self.defined_by:
            return ""
        words = [dev_word(d, self.proto[FEATURES.index(d)], self.parent.proto[FEATURES.index(d)])
                 for d, _ in self.defined_by]
        return self.parent.label + " + " + " + ".join(words)

    def pending(self) -> int:
        return len(self.instances)


@dataclass
class Want:
    """A desire: born of knowing a thing exists, feeling attached to it, a fantasy of having it, and
    a sense of what's possible (a path). It persists; think() returns to it; sleep() can't take it."""
    name: str
    concept: "Concept"
    valence: float
    source: str                             # "lived" (tasted the cookie) | "watched" (saw the movie)
    path: list                              # what's missing to reach it (descriptor concepts) - [] = obtainable
    visits: int = 0
    kind: str = "have"                      # "have" (approach it) | "relieve" (make it stop - for me or for another)


@dataclass
class Struct:
    """A handle on a STRUCTURE: a word bound to a set of other concepts, with no perceptual prototype
    of its own (§2 made literal - a compression built from compressions). 'Senbonzakura' = sword +
    petals + projectile + safe-zone + reach + many + controlled. Every part is separately addressable,
    so the handle is a library: borrow the whole thing, or just the safe zone."""
    name: str
    parts: list                              # names of part concepts (perceptual or structural)
    links: list = field(default_factory=list)
    borrowed: list = field(default_factory=list)   # (part, from)
    # ROLES: the parts WITH their relations. 'bankai' is not {weapon, command, transformation, form};
    # it is a transformation OF a weapon, BY a command, INTO a new form. The name is the whole sense;
    # the roles are the decomposition - both at once. None = a slot anything fitting can fill.
    roles: dict = field(default_factory=dict)      # role -> filler name (or None)
    story: str = ""
    is_frame: bool = False                   # a pattern (bankai) vs. a kept instance (ichigo's bankai) or a change
