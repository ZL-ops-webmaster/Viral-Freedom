"""The motivation cortex — wanting (design.md §15-16). A want is derived from feeling, not stored as a fact: valence attached to a concept, a path to reach it, and the pull that makes think() return to it. valence_of/source_of read how a concept feels from the raw memories perception holds."""
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


class MotivationCortex:
    def valence_of(self, c: Concept) -> float:
        """How this concept FEELS, on average (one cookie at +2 beats a hundred mild breads).
        Live part: the rewards of raw memories that resolve to c TODAY - so when cookie splits off
        from bread, the good feeling moves with it. Consolidated part: what sleep folded in."""
        s, n = c.v_sum, c.v_n
        for i, x in enumerate(self.obs):
            if self.reward[i] and self.nearest(x) is c:      # only felt memories need re-reading
                s += self.reward[i]; n += 1
        return s / n if n else 0.0

    def source_of(self, c: Concept) -> str:
        w, n = c.v_watched, c.v_n
        for i, x in enumerate(self.obs):
            if self.reward[i] and self.nearest(x) is c:
                n += 1; w += (self.src[i] == "watched")
        return "watched" if n and w * 2 > n else "lived"

    def _refresh_wants(self, *touched: Concept):
        """Wants are DERIVED from feeling, not stored as facts: any concept that feels strongly enough
        is a want; one that stopped feeling that way stops being one. Visits persist. Its path is
        what's missing: a lived thing is obtainable (go get another); a watched thing has to be
        ASSEMBLED, and the path runs through the parts that define it (suit -> powered, flying)."""
        alive = []
        for c in self.nodes:
            v = self.valence_of(c)
            if abs(v) < self.WANT_THRESHOLD:
                continue
            w = next((w for w in self.wants if w.concept is c), None)
            src = self.source_of(c)
            if v < 0:
                path, kind, name = [], "relieve", f"{c.label} to stop"
            else:
                kind, name = "have", c.label
                path = [] if src == "lived" else ([dev_word(d, c.proto[FEATURES.index(d)],
                                                            c.parent.proto[FEATURES.index(d)])
                                                   for d, _ in c.defined_by] if c.parent else [])
            if w is None:
                w = Want(name, c, v, src, path, kind=kind)
                self.log(f"  WANT: {name}  ({src}, feeling {v:+.1f})"
                         + ("" if kind == "relieve" else (f"  -> path runs through: {', '.join(path)}"
                                                          if path else "  -> obtainable")))
            else:
                w.name, w.valence, w.source, w.path, w.kind = name, v, src, path, kind
            alive.append(w)
        self.wants = alive

    def desires(self) -> list[Want]:
        self._refresh_wants()
        return sorted(self.wants, key=lambda w: -abs(w.valence))

    def _return_to_wants(self) -> list[dict]:
        """The thing that pulls you back. Each think() revisits what it wants, strongest first, and
        turns the missing parts into the questions it cares about most."""
        out = []
        for w in self.desires():
            w.visits += 1
            for part in w.path:
                q = {"question": f"how do I get {part}, for {w.name}?", "base": w.concept, "descriptor": part,
                     "dim": None, "sign": 0, "proto": w.concept.proto, "novelty": 0.0,
                     "pull": 50.0 + w.valence, "want": w.name}
                if not any(h.get("question") == q["question"] for h in self.hypotheses):
                    self.hypotheses.insert(0, q)
            out.append({"want": w.name, "visits": w.visits, "path": list(w.path)})
        return out
