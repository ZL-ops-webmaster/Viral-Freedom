"""The thinking cortex (design.md §5). Running the tree WITHOUT the world: predict unseen features, imagine combinations, wonder up questions, let expectation steer perception, mint from one example, and the idle think() loop that feeling directs."""
from __future__ import annotations

from collections import Counter
from typing import Optional
import math
import numpy as np

from core import (
    FEATURES, DESCRIPTOR_WORDS, dev_word,
    K_TOL, SPREAD_FLOOR, NODE_COST, MIN_RECUR, OVERLAP,
    Concept, Want, Struct,
)


class ThinkingCortex:
    def descriptor_concepts(self) -> dict:
        """A descriptor IS a concept: the reusable delta along one dimension that recurring splits
        share. 'big' = what dog->bear and dog->wolf have in common on the size axis. Once minted it
        transfers to any branch (analogy)."""
        acc: dict = {}
        for c in self.nodes:
            if c.parent is None:
                continue
            for dim, sgn in c.defined_by:
                i = FEATURES.index(dim)
                acc.setdefault((dim, sgn), []).append((c.proto[i] - c.parent.proto[i], c.label))
        out = {}
        for (dim, sgn), rows in acc.items():
            name = (("more-legs" if sgn > 0 else "fewer-legs") if dim == "leg_count"
                    else DESCRIPTOR_WORDS[dim][1 if sgn > 0 else 0])
            out[name] = {"dim": dim, "sign": sgn, "delta": float(np.mean([d for d, _ in rows])),
                         "from": [lab for _, lab in rows]}
        return out

    def predict(self, partial) -> tuple[Concept, np.ndarray, float]:
        """partial: feature vector with NaN for unknown dims. Returns (concept it expects, the
        filled-in vector, how well the known part fits). This is inference: the unseen dims are
        inherited from the concept, before the world has shown them."""
        partial = np.asarray(partial, float)
        known = ~np.isnan(partial)
        def fit(c):
            return float(np.sqrt(np.mean(((partial[known] - c.proto[known]) / c.spread[known]) ** 2)))
        c = min(self.nodes, key=fit)
        filled = partial.copy(); filled[~known] = c.proto[~known]
        return c, filled, fit(c)

    def imagine(self, concept: Concept, names: list) -> tuple[np.ndarray, Concept, float]:
        """Apply minted descriptor concepts to a concept -> a prototype of something that may not
        exist. Returns (imagined vector, nearest known concept, distance). Far from everything =
        a genuinely new idea."""
        D = self.descriptor_concepts()
        p = concept.proto.copy()
        for n in names:
            d = D[n]; p[FEATURES.index(d["dim"])] += d["delta"]
        near = self.nearest(p)
        return p, near, near.distance(p)

    def wonder(self, top: int = 6) -> list[dict]:
        """Unprompted. For every concept and every minted descriptor: imagine the combination.
        If nothing known covers it, that's a hypothesis the system now wants the world to answer."""
        D = self.descriptor_concepts()
        hyps = []
        for c in self.nodes:
            for name, d in D.items():
                if (d["dim"], d["sign"]) in c.defined_by:          # that's just how c is already defined
                    continue
                # NOTE: a sign-reversal of c's own definition (a "big" cat when cat = small dog) is a
                # REAL question, not a redundancy - that's exactly what a lion is. Novelty filters the rest.
                p, near, dist = self.imagine(c, [name])
                if dist > self.NOVEL:
                    # feeling directs attention: a branch with things it can't explain gets its
                    # questions pushed to the front (unease -> curiosity about THAT branch)
                    pull = dist + 0.5 * len(c.instances)
                    hyps.append({"question": f"is there a {name} {c.label}?", "base": c, "descriptor": name,
                                 "dim": d["dim"], "sign": d["sign"], "proto": p, "novelty": dist, "pull": pull})
        hyps.sort(key=lambda h: -h["pull"])
        self.hypotheses = hyps
        return hyps[:top]

    def _expectation(self, x: np.ndarray):
        """Top-down perception. If an open question predicts this observation (it deviates from the
        question's base concept in exactly the hypothesised way, and the base is nearly as close as
        anything else), the thought wins the tie. Returns (base, deviations, hypothesis) or None."""
        if not self.hypotheses:
            return None
        near = self.nearest(x); dn = near.distance(x)
        bottom_up = len(near.deviations(x))      # what the world costs to describe WITHOUT the thought
        best = None
        for h in self.hypotheses:
            base = h["base"]
            devs = base.deviations(x)
            # a thought may steer perception only if it does NOT make the world costlier to describe
            # (compression gates expectation too); ties go to the thought - that is what it is for
            matches = ((h["dim"], h["sign"]) in devs) if h["dim"] is not None else bool(devs)  # a kind of the head
            if (matches and len(devs) <= bottom_up
                    and base.distance(x) <= self.EXPECT_SLACK * dn):
                # among the questions this observation could answer, take the one whose base
                # explains it most CHEAPLY (fewest descriptors) - compression picks the parent
                key = (len(devs), base.distance(x))
                if best is None or key < best[0]:
                    best = (key, base, devs, h)
        return best[1:] if best else None

    def _mint(self, parent: Concept, x: np.ndarray, devs: tuple, word: Optional[str]) -> Concept:
        """Make a concept from ONE example. Only allowed when the structure already predicted it
        (an answered hypothesis): the question supplied the evidence the example alone lacks."""
        child = Concept(self._next, x.copy(), np.zeros_like(x), n=1, parent=parent, defined_by=devs,
                        prior=parent.spread.copy())           # inherits how much its parent varies
        self._next += 1
        if word:
            child.word_counts[word] = MIN_RECUR       # the name arrived with the answer
            child.called.add(word)
        parent.children.append(child); self.nodes.append(child)
        if word:
            self._link_compound(child, word)
        return child

    def think(self, top: int = 6) -> dict:
        """One idle 'thought'. Nothing asked for it. Feeling decides what to do with the time:
        uneasy -> consolidate what's unexplained; otherwise -> wonder (curiosity). Then the pulses fade."""
        _, mood_before = self.feel()
        made = self._generalize_all()                         # the day's words reorganise the tree
        if self.feeling.unease >= 1.0:
            made += self.idle()                               # settle the debt
        qs = self.wonder(top)
        returned = self._return_to_wants()                    # then: back to what it wants
        qs = self.hypotheses[:top]
        self.feeling.fade()                                                   # pulses fade
        _, mood_after = self.feel()
        return {"mood_before": mood_before, "consolidated": made, "descriptors": self.descriptor_concepts(),
                "questions": qs, "returned_to": returned, "mood_after": mood_after}
