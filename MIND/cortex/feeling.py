"""The feeling cortex — valence that directs attention (design.md §5b).

Feeling is the derivative of compression. This region owns the two PULSES:
  - surprise      — a prediction missed; the jolt that opens learning.
  - satisfaction  — the world got cheaper to describe; understanding felt.

It does NOT own unease or curiosity as stored drives. Those are read off
signals the other cortices hand in when feeling is asked how it feels:
  - unexplained memories   (from perception)
  - open questions + pulls (from thinking)
That hand-in is the point of the split: feeling can only know the whole mood
by being TOLD what the other regions are holding. The communication is explicit
instead of being a shared dictionary everyone pokes.
"""

from __future__ import annotations

from typing import List, Tuple

CALM = 0.3          # below this on every channel, it is simply calm


class Feeling:
    def __init__(self):
        self.surprise = 0.0
        self.satisfaction = 0.0
        self.unease = 0.0        # last value computed from the signals handed in
        self.curiosity = 0.0     # "
        self.last = ""           # the last mood read, as a sentence

    # -- signals in: the other cortices push these ------------------------

    def jolt(self, amount: float) -> None:
        """A prediction missed by `amount`. Blend it into the surprise pulse."""
        self.surprise = 0.5 * self.surprise + 0.5 * amount

    def satisfy(self, amount: float) -> None:
        """Something clicked / was rewarding: add to the satisfaction pulse."""
        self.satisfaction += amount

    def fade(self) -> None:
        """After a thought, the pulses decay (they are momentary, not standing)."""
        self.surprise *= 0.5
        self.satisfaction *= 0.5

    # -- read out ---------------------------------------------------------

    def read(self) -> dict:
        return {"surprise": self.surprise, "satisfaction": self.satisfaction,
                "unease": self.unease, "curiosity": self.curiosity}

    def feel(self, unexplained: int, hypotheses: List[dict]) -> Tuple[dict, str]:
        """Compute the current mood. unease = compression debt (things it can't
        explain + open questions); curiosity = pull of the best open question;
        surprise/satisfaction are the recent pulses. Returns (affect, sentence).

        `unexplained` comes from perception, `hypotheses` from thinking -- the
        signals this cortex needs but does not itself hold."""
        self.unease = unexplained + 0.5 * len(hypotheses)
        self.curiosity = max((h["pull"] for h in hypotheses), default=0.0)
        a = self.read()
        cands = [
            ("surprised", a["surprise"], f"something didn't fit (surprise {a['surprise']:.1f})"),
            ("satisfied", a["satisfaction"] / 3, f"things clicked (+{a['satisfaction']:.0f})"),
            ("uneasy", a["unease"] / 2, f"{unexplained} things I can't explain, {len(hypotheses)} open questions"),
            ("curious", a["curiosity"] / 2, (hypotheses[0]["question"] if hypotheses else "nothing to ask")),
        ]
        name, _, detail = max(cands, key=lambda t: t[1])
        if max(t[1] for t in cands) < CALM:
            name, detail = "calm", "everything fits"
        self.last = f"{name}: {detail}"
        return a, self.last
