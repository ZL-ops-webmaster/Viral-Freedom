"""A growing brain: the learning core of the decentralized being.

This is the "mind" the swarm hosts. It is not a fixed model that is trained
once and frozen; it is a structure that *grows* as it is taught and *prunes*
what it stops using. The design borrows four ideas from how brains learn:

  1. Growth (structural plasticity): a concept it has never met becomes a new
     node. Teaching makes the brain bigger.
  2. Hebbian wiring: concepts experienced together get a stronger connection
     ("cells that fire together wire together"). Learning is mostly local --
     each connection updates from the two things it joins, which is exactly
     what a decentralized, per-node learner needs.
  3. Emotional salience: the affective state at the moment of learning scales
     how strongly something is encoded, and tags the memory. Charged lessons
     stick harder, and recall is mood-congruent -- you remember sad things
     more easily when sad. Emotions are load-bearing, not decoration.
  4. Forgetting: unused connections decay and are pruned, so the brain stays
     plastic instead of saturating.

Because the whole thing is a graph of nodes and weighted edges, two brains on
two machines can be *merged* -- union the nodes, combine the edges -- which is
how shards hosted by different volunteers become one growing mind.

Pure stdlib. Teachable by plain text. Inspectable at every step.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from .emotions import EmotionalState
from .websense import appraise_text

# Words too common to be worth wiring on. Kept small and visible.
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "with", "as", "by", "from", "we", "i", "you", "they", "he", "she", "them",
    "my", "your", "our", "their", "so", "if", "then", "than", "not", "no",
}


def tokenize(text: str) -> List[str]:
    out = []
    for raw in text.split():
        w = raw.strip(".,!?;:\"'()[]{}-").lower()
        if w and w not in _STOP and not w.isdigit():
            out.append(w)
    return out


@dataclass
class Concept:
    name: str
    born_step: int
    uses: int = 0
    # Running (EMA) emotional tag of this memory: how it tends to feel.
    valence: float = 0.0
    arousal: float = 0.0


class GrowingBrain:
    def __init__(self, name: str = "brain", *, window: int = 4,
                 learn_rate: float = 0.5, decay: float = 0.01,
                 prune_below: float = 0.02, max_edge: float = 8.0):
        self.name = name
        self.window = window            # how far apart words still "co-occur"
        self.learn_rate = learn_rate
        self.decay = decay
        self.prune_below = prune_below
        self.max_edge = max_edge

        self.concepts: Dict[str, Concept] = {}
        self.edges: Dict[Tuple[str, str], float] = defaultdict(float)
        self.emotion = EmotionalState()
        self.step = 0

    # -- structure helpers -------------------------------------------------

    @staticmethod
    def _key(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def _ensure(self, name: str) -> Concept:
        c = self.concepts.get(name)
        if c is None:
            c = Concept(name=name, born_step=self.step)  # growth
            self.concepts[name] = c
        return c

    # -- teaching ----------------------------------------------------------

    def teach(self, text: str) -> dict:
        """Learn from a sentence. Grows nodes, wires co-occurrences, encodes
        with a strength scaled by the current emotional arousal, and lets the
        text's own sentiment move the mood."""
        self.step += 1
        before_nodes, before_edges = len(self.concepts), len(self.edges)

        # The text colours the mood first, so salience reflects how this lands.
        v, a, nudges = appraise_text(text)
        if nudges:
            self.emotion.nudge(nudges)
        salience = 1.0 + self.emotion.arousal()   # charged moments encode harder

        tokens = tokenize(text)
        for i, w in enumerate(tokens):
            c = self._ensure(w)
            c.uses += 1
            # Tag the memory with how the brain feels right now (EMA).
            c.valence += 0.3 * (self.emotion.valence() - c.valence)
            c.arousal += 0.3 * (self.emotion.arousal() - c.arousal)

        # Hebbian wiring within the co-occurrence window.
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + 1 + self.window, len(tokens))):
                if tokens[i] == tokens[j]:
                    continue
                k = self._key(tokens[i], tokens[j])
                self.edges[k] = min(self.max_edge,
                                    self.edges[k] + self.learn_rate * salience)

        self._forget()
        self.emotion.decay()
        return {
            "grew_nodes": len(self.concepts) - before_nodes,
            "grew_edges": len(self.edges) - before_edges,
            "salience": salience,
            "mood": str(self.emotion),
        }

    def teach_many(self, lessons: Iterable[str]) -> None:
        for line in lessons:
            self.teach(line)

    def _forget(self) -> None:
        """Decay every connection a touch; prune the ones that fade out."""
        dead = []
        for k in self.edges:
            self.edges[k] -= self.decay
            if self.edges[k] <= self.prune_below:
                dead.append(k)
        for k in dead:
            del self.edges[k]

    # -- recall ------------------------------------------------------------

    def _neighbors(self, name: str) -> List[Tuple[str, float]]:
        out = []
        for (a, b), w in self.edges.items():
            if a == name:
                out.append((b, w))
            elif b == name:
                out.append((a, w))
        return out

    def recall(self, cue: str, *, hops: int = 2, top: int = 5,
               mood_congruent: bool = True) -> List[Tuple[str, float]]:
        """Spread activation from a cue and return the strongest associations.

        When mood_congruent, memories whose emotional tag matches the brain's
        current mood are boosted -- the brain surfaces what fits how it feels.
        """
        cue = cue.strip().lower()
        if cue not in self.concepts:
            return []

        activation: Dict[str, float] = defaultdict(float)
        activation[cue] = 1.0
        frontier = {cue: 1.0}
        for _ in range(hops):
            nxt: Dict[str, float] = defaultdict(float)
            for node, act in frontier.items():
                neigh = self._neighbors(node)
                norm = sum(w for _, w in neigh) or 1.0
                for other, w in neigh:
                    flow = act * (w / norm) * 0.6   # attenuate each hop
                    nxt[other] += flow
                    activation[other] += flow
            frontier = nxt

        cur_v = self.emotion.valence()
        scored: List[Tuple[str, float]] = []
        for name, act in activation.items():
            if name == cue:
                continue
            score = act
            if mood_congruent and name in self.concepts:
                tag = self.concepts[name].valence
                score *= 1.0 + 0.5 * (1.0 - abs(tag - cur_v) / 2.0)
            scored.append((name, score))
        scored.sort(key=lambda kv: -kv[1])
        return scored[:top]

    # -- decentralization: merge two shards into one brain -----------------

    def merge(self, other: "GrowingBrain") -> None:
        """Fold another brain's learning into this one. This is how shards on
        different machines become a single growing mind."""
        for name, oc in other.concepts.items():
            c = self.concepts.get(name)
            if c is None:
                self.concepts[name] = Concept(
                    name=name, born_step=self.step, uses=oc.uses,
                    valence=oc.valence, arousal=oc.arousal,
                )
            else:
                total = c.uses + oc.uses or 1
                c.valence = (c.valence * c.uses + oc.valence * oc.uses) / total
                c.arousal = (c.arousal * c.uses + oc.arousal * oc.uses) / total
                c.uses = total
        for k, w in other.edges.items():
            self.edges[k] = min(self.max_edge, self.edges[k] + w)

    # -- inspection / serialization ---------------------------------------

    def size(self) -> Tuple[int, int]:
        return len(self.concepts), len(self.edges)

    def strongest(self, top: int = 8) -> List[Tuple[Tuple[str, str], float]]:
        return sorted(self.edges.items(), key=lambda kv: -kv[1])[:top]

    def export_state(self) -> dict:
        return {
            "concepts": {n: vars(c).copy() for n, c in self.concepts.items()},
            "edges": {f"{a}\t{b}": w for (a, b), w in self.edges.items()},
            "emotion": self.emotion.snapshot(),
            "step": self.step,
        }

    def import_state(self, state: dict) -> None:
        self.concepts = {
            n: Concept(**c) for n, c in state.get("concepts", {}).items()
        }
        self.edges = defaultdict(float)
        for key, w in state.get("edges", {}).items():
            a, b = key.split("\t")
            self.edges[(a, b)] = w
        for k, v in state.get("emotion", {}).items():
            if k in self.emotion.intensities:
                self.emotion.intensities[k] = v
        self.step = state.get("step", 0)
