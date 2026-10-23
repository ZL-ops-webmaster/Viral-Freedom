"""Grammar adapter — learned sentence patterns, not a hardcoded parser.

The old text layer (talk.py) understood a sentence by fixed rules: strip a
determiner, word 1 is the subject, word 2 must BE a known frame, fill the rest
by position. A dozen regexes. It cannot meet a sentence shaped in a way nobody
wrote a rule for.

This layer learns the shapes instead, by the same move the brain already uses
to induce a frame from episodes (design.md §23): line up a few examples, and a
position whose word VARIES becomes a slot, a position whose word STAYS becomes
fixed structure. Point that at surface word order and you get grammar.

Zion's spec (the Japanese analogy): you cannot speak a language from knowing
each word alone; you hear sentences strung together a few times, then you can
produce the shape -- even before you fully understand it -- and the meaning
arrives later from context. So:

  - A pattern is LEARNED from a handful of examples, never hardcoded.
  - parse() binds the slots even when the filler words are UNKNOWN (speak the
    shape before you understand it). Meaning is deferred to the brain's own
    intuit() -- figured out later from the company a word keeps.
  - produce() runs a pattern backwards: given a meaning, say it.

Prepositions and function words need no special code: they are simply the
positions that never vary, so induction keeps them as fixed anchors for free.

Pure stdlib. The brain is not imported or modified; integration is by its
public methods only (see demo_grammar.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Kept identical to talk.py's sets so the two layers agree on function words.
DETS = {"a", "an", "the", "my", "your", "his", "her", "their", "some"}


@dataclass
class Tok:
    """One position in a pattern."""
    kind: str          # "lit" (fixed word) | "det" (optional determiner) | "slot"
    value: str = ""    # the word for lit; the role name for slot; "" for det


@dataclass
class Pattern:
    frame: str                      # the brain frame this shape means, e.g. "give"
    toks: List[Tok]
    examples: int = 1
    heard: List[str] = field(default_factory=list)

    def slots(self) -> List[str]:
        return [t.value for t in self.toks if t.kind == "slot"]

    def shape(self) -> str:
        parts = []
        for t in self.toks:
            parts.append(t.value if t.kind == "lit"
                         else "DET" if t.kind == "det" else f"[{t.value}]")
        return " ".join(parts)


def _words(sentence: str) -> List[str]:
    return sentence.lower().replace(",", "").split()


class Grammar:
    def __init__(self):
        self.patterns: List[Pattern] = []

    # -- exposure: learn a shape from an example --------------------------

    def teach(self, sentence: str, frame: str, fillers: Dict[str, str]) -> Pattern:
        """Hear one example and fold it into what's known.

        `fillers` says which words are the meaning's roles (giver=ichigo, ...).
        Everything else in the sentence is structure: determiners become
        optional det-slots, all other words become fixed anchors (this is how
        'gave', 'with', 'a' fall out as grammar without being special-cased).
        """
        val2role = {v.lower(): r for r, v in fillers.items()}
        words = _words(sentence)
        toks: List[Tok] = []
        i = 0
        while i < len(words):
            w = words[i]
            # A determiner right before a filler is part of that slot (slots
            # swallow a leading determiner), so it never becomes its own token.
            # This keeps 'ichigo gave...' and 'the knight gave...' the SAME
            # shape -- determiners are optional, not a position of their own.
            if w in DETS and i + 1 < len(words) and words[i + 1] in val2role:
                toks.append(Tok("slot", val2role[words[i + 1]]))
                i += 2
                continue
            if w in val2role:
                toks.append(Tok("slot", val2role[w]))
            elif w in DETS:
                toks.append(Tok("det"))
            else:
                toks.append(Tok("lit", w))
            i += 1

        for p in self.patterns:              # merge into a matching known shape
            if p.frame == frame and self._mergeable(p.toks, toks):
                p.toks = self._merge(p.toks, toks)
                p.examples += 1
                p.heard.append(sentence)
                return p
        p = Pattern(frame, toks, 1, [sentence])
        self.patterns.append(p)
        return p

    @staticmethod
    def _mergeable(a: List[Tok], b: List[Tok]) -> bool:
        # Positional alignment only (same honest limit as the brain's frame
        # induction): same length, same slot roles in the same places.
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if (x.kind == "slot") != (y.kind == "slot"):
                return False
            if x.kind == "slot" and x.value != y.value:
                return False
        return True

    @staticmethod
    def _merge(a: List[Tok], b: List[Tok]) -> List[Tok]:
        out = []
        for x, y in zip(a, b):
            if x.kind == "slot":
                out.append(x)
            elif x.kind == "lit" and y.kind == "lit" and x.value == y.value:
                out.append(x)                       # a constant word -> stays fixed
            else:
                out.append(Tok("det"))              # it varied -> optional/function
        return out

    # -- parse: bind a sentence's slots, known words or not ---------------

    def parse(self, sentence: str) -> Optional[dict]:
        """Match the sentence to the best-fitting learned pattern.

        Returns {"frame", "fillers", "shape"} or None. Fillers are bound even
        when their words mean nothing yet -- that is speaking before
        understanding; the brain fills meaning in afterwards via intuit().
        """
        words = _words(sentence)
        best = None
        for p in self.patterns:
            fills = self._match(p.toks, words)
            if fills is None:
                continue
            anchors = sum(1 for t in p.toks if t.kind == "lit")   # more fixed words = surer
            key = (anchors, p.examples)
            if best is None or key > best[0]:
                best = (key, p, fills)
        if best is None:
            return None
        _, p, fills = best
        return {"frame": p.frame, "fillers": fills, "shape": p.shape()}

    @staticmethod
    def _match(pat: List[Tok], words: List[str]) -> Optional[Dict[str, str]]:
        i = 0
        fills: Dict[str, str] = {}
        for t in pat:
            if t.kind == "lit":
                if i < len(words) and words[i] == t.value:
                    i += 1
                else:
                    return None
            elif t.kind == "det":
                if i < len(words) and words[i] in DETS:
                    i += 1                              # optional
            else:  # slot
                if i < len(words) and words[i] in DETS and i + 1 < len(words):
                    fills[t.value] = words[i + 1]       # swallow a leading determiner
                    i += 2
                elif i < len(words) and words[i] not in DETS:
                    fills[t.value] = words[i]
                    i += 1
                else:
                    return None
        return fills if i == len(words) else None       # strict: consume all

    # -- produce: run a pattern backwards ---------------------------------

    def produce(self, frame: str, fillers: Dict[str, str]) -> Optional[str]:
        """Say a meaning. Picks a learned pattern for the frame whose slots the
        fillers cover, and emits it."""
        best = None
        for p in self.patterns:
            if p.frame == frame and all(s in fillers for s in p.slots()):
                if best is None or p.examples > best.examples:
                    best = p
        if best is None:
            return None
        out = []
        for t in best.toks:
            if t.kind == "lit":
                out.append(t.value)
            elif t.kind == "det":
                out.append("the")
            else:
                out.append(fillers[t.value])
        return " ".join(out)

    # -- autonomous induction: slots from raw sentences, no labels --------

    def observe(self, sentences: List[str], min_group: int = 2) -> List["Pattern"]:
        """Induce patterns from RAW sentences with NO roles marked.

        The Japanese analogy taken all the way: nobody says which word is the
        giver. Strip the optional function words, group sentences that put the
        same verb in the same place, and read the columns -- a column whose
        word never changes is a fixed anchor, a column whose word varies is a
        slot. Slots are auto-named (s0, s1, ...) exactly as the brain names an
        unnarrated frame position step-k (design.md §23); the verb names the
        frame. Meaning (which slot is the giver) is attached later, by
        narration or by the brain's intuit() -- not needed to learn the shape.
        """
        from collections import defaultdict
        buckets: Dict[tuple, List[tuple]] = defaultdict(list)
        for sent in sentences:
            content = [w for w in _words(sent) if w not in DETS]
            if len(content) >= 2:                       # need a subject and a verb
                buckets[(len(content), content[1])].append((sent, content))

        made: List[Pattern] = []
        for (length, verb), group in buckets.items():
            if len(group) < min_group:
                continue                                # one example is not a shape
            toks: List[Tok] = []
            slot_k = 0
            for col in range(length):
                words_here = {c[col] for _, c in group}
                if len(words_here) == 1:                # never varied -> anchor
                    toks.append(Tok("lit", group[0][1][col]))
                else:                                   # varied -> a slot
                    toks.append(Tok("slot", f"s{slot_k}"))
                    slot_k += 1
            shape = Pattern(verb, toks).shape()
            existing = next((p for p in self.patterns
                             if p.frame == verb and p.shape() == shape), None)
            if existing:
                existing.examples += len(group)
                existing.heard.extend(s for s, _ in group)
            else:
                p = Pattern(verb, toks, len(group), [s for s, _ in group])
                self.patterns.append(p)
                made.append(p)
        return made

    def name_slots(self, frame: str, names: Dict[str, str]) -> bool:
        """Attach meaning after the fact: rename a frame's auto slots (s0->giver).
        This is narration arriving later, the shape already learned."""
        changed = False
        for p in self.patterns:
            if p.frame != frame:
                continue
            for t in p.toks:
                if t.kind == "slot" and t.value in names:
                    t.value = names[t.value]
                    changed = True
        return changed

    # -- inspection --------------------------------------------------------

    def known(self) -> List[str]:
        return [f"{p.frame}: {p.shape()}  (x{p.examples})" for p in self.patterns]
