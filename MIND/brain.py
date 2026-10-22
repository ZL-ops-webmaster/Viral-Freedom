"""B.R.A.I.N. — Basic Rational And Intelligent Neuronetwork. Prototype v0.1.

Implements design.md §2-§3b, §7 on a simple perceptual substrate (feature vectors):

  concept        = a SENSE: a compressed region (prototype + per-dim tolerance) in perceptual space
  recognize      = ping the nearest concept, normalised by its tolerance
  explain(x)     = nearest concept  ∘  descriptors   (one descriptor per feature outside tolerance)
  learn(x)       = fits → sharpen the sense (one-example update)
                   doesn't fit → store "concept + descriptors"; that bundle is a candidate split
  split          = fires when the SAME KIND of bundle recurs (grouped by overlap, so a flickering
                   borderline feature doesn't break recurrence) and a new node is the shorter total
                   description:  cost(stay) = Σ(1 + |bundle_i|)   vs   cost(new) = NODE + |core| + n
  word handle    = any word co-occurring with a concept binds to it by recurrence (§7); a handle is a
                   door into the node, never the node itself
  idle()         = unprompted consolidation: re-run the compression decision over everything stored

Everything runs on one principle: the shortest description wins.
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
    global FEATURES, DESCRIPTOR_WORDS
    FEATURES = list(features); DESCRIPTOR_WORDS = dict(descriptor_words)


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


# ---- the mind ----------------------------------------------------------------------------
class Brain:
    def __init__(self, log=print):
        self.nodes: list[Concept] = []
        self._next = 1
        self.log = log
        self.tokens_spent = 0               # running cost of every description it has produced
        self.hypotheses: list[dict] = []    # open questions generated by thinking (see wonder())
        # FEELING = the derivative of compression (design.md §5b). Valence that directs attention.
        self.affect = {"surprise": 0.0, "satisfaction": 0.0, "unease": 0.0, "curiosity": 0.0}
        self.last_felt = ""
        # CONTEXT = working memory: what is lit up right now, and what tends to show up together.
        # This is what resolves a DOUBLE (one handle, several concepts): "bat" after a ball is the club.
        # Company is NOT stored as node ids - it is re-read through the CURRENT tree from the raw
        # observations, so every split retroactively corrects who was with whom.
        self.obs: list[np.ndarray] = []     # every observation, in order
        self.said: list[str] = []           # what it CALLED each observation at the time (its reading then)
        self.weight: list[float] = []       # PRIORITY of each raw memory: surprise, significance, re-reads
        self.reward: list[float] = []       # how each raw memory FELT (0 = no feeling data, not "neutral")
        self.src: list[str] = []            # "lived" | "watched" - where that feeling came from
        self.recent: list[int] = []         # indices of the last few observations (the window)
        self.WINDOW = 3
        self.forgotten = 0                  # how many raw memories sleep() has let go
        self.wants: list[Want] = []         # what it desires (see Want); the "me" in "matters to me"
        self.structs: dict[str, Struct] = {}   # structural concepts (see Struct), kept apart from the perceptual tree
        self.role_fills: Counter = Counter()   # (role, filler) -> times seen: what has played what
        self.same: dict = {}                   # simple equivalences: transform = change
        self.episodes: dict = {}               # name -> list of ordered experiences (sequences of fillers)
        self.episode_labels: dict = {}         # name -> what the narration called each position
        self.WATCHED_DISCOUNT = 0.5         # feeling borrowed from watching counts half of living it

    # -- recognition ----------------------------------------------------------------------
    def nearest(self, x: np.ndarray) -> Concept:
        return min(self.nodes, key=lambda c: c.distance(x))

    def explain(self, x: np.ndarray) -> tuple[Concept, tuple]:
        c = self.nearest(x)
        return c, c.deviations(x)

    def describe(self, x: np.ndarray) -> str:
        c, dev = self.explain(x)
        words = [c.label] + [dev_word(d, x[FEATURES.index(d)], c.proto[FEATURES.index(d)]) for d, _ in dev]
        return " + ".join(words)

    def tokens(self, x: np.ndarray) -> int:
        _, dev = self.explain(x)
        return 1 + len(dev)

    def _record(self, x: np.ndarray) -> tuple:
        """Log the observation; return the window of what came just before it (its company)."""
        ctx = tuple(self.recent)
        self.obs.append(x)
        self.recent = [len(self.obs) - 1] + self.recent[: self.WINDOW - 1]
        return ctx

    def active_nodes(self) -> list[Concept]:
        """What is in mind right now: the concepts the last few observations resolve to - TODAY."""
        out = []
        for i in self.recent:
            c = self.nearest(self.obs[i])
            if c not in out: out.append(c)
        return out

    # -- learning ---------------------------------------------------------------------------
    def learn(self, x: np.ndarray, word: Optional[str] = None, reward: float = 0.0) -> str:
        """reward = how encountering this FELT, from the world (the cookie tastes good: +). It binds
        to the concept the observation resolves to, and is where wanting starts."""
        return self._experience(x, word, reward, source="lived")

    def watch(self, x: np.ndarray, word: Optional[str] = None, reward: float = 0.0) -> str:
        """Seeing it, not living it (the Iron Man movie). You still get the concept; the feeling is
        borrowed, so it binds at a discount - but it binds, and that is enough to want."""
        return self._experience(x, word, reward * self.WATCHED_DISCOUNT, source="watched")

    def _experience(self, x, word, reward, source) -> str:
        x = np.asarray(x, float)
        s0, n0 = self.affect["surprise"], len(self.nodes)
        out = self._learn(x, word)
        if reward and abs(reward) >= self.WANT_THRESHOLD and len(self.nodes) == n0:
            # a strong feeling is evidence enough: a thing that felt THAT different earns a concept
            # from ONE example - one bad fall teaches "falling", one cookie teaches "cookie".
            # (significance substitutes for recurrence, exactly as an answered question does)
            c = self.nearest(x); dev = c.deviations(x)
            held = next((t for t in c.instances if t[0] is x), None)
            if dev and held is not None:
                c.instances = [t for t in c.instances if t is not held]
                child = self._mint(c, x, dev, word)
                out = child.label
                self.log(f"  FELT {reward:+.1f}: that's not just {c.label} -> minted {child.label} from ONE example")
        self.said.append(out)                                # remember what it called this, back then
        # priority at the moment of seeing it: a thing that fit perfectly and changed nothing is a
        # washer breaking at work; a thing that surprised it or made a concept is worth keeping
        w = 1.0 + max(0.0, self.affect["surprise"] - s0) * 2.0 + (3.0 if len(self.nodes) > n0 else 0.0)
        self.weight.append(w + abs(reward))                  # things that made you feel are worth keeping
        self.reward.append(reward); self.src.append(source)  # the feeling stays WITH the raw memory
        c = self.nearest(x)
        if reward:
            self.affect["satisfaction"] += reward
            self._refresh_wants(c)
        elif abs(self.valence_of(c)) >= self.WANT_THRESHOLD:
            # RESONANCE / EMPATHY: a concept it has already FELT fires that feeling again on
            # recognition - at full strength, because it was earned by living it, not borrowed. And
            # no reward reached ME, so this is happening to someone else. That is how a kid who has
            # fallen knows the other kid hurts - and the one who never fell walks past.
            v = self.valence_of(c)
            self.affect["satisfaction"] += v
            self.weight[-1] += abs(v)
            if v < 0:
                self.log(f"  FEELS: {c.label} - I know that hurts.  -> HELPS (wants it to stop)")
            else:
                self.log(f"  FEELS: {c.label} - I know that's good.  -> wants it")
            self._refresh_wants(c)
        return out

    # -- feeling about a concept: re-read through the current tree, plus what sleep consolidated ---
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

    # -- wanting -----------------------------------------------------------------------------
    WANT_THRESHOLD = 1.0

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

    def _learn(self, x: np.ndarray, word: Optional[str] = None) -> str:
        x = np.asarray(x, float)
        ctx = self._record(x)                                # what was in mind when this was seen
        if not self.nodes:                                   # the one expensive grounded sense
            root = Concept(self._next, x.copy(), np.zeros_like(x)); self._next += 1
            self.nodes.append(root)
            if word: root.word_counts[word] += 1
            self.tokens_spent += 1
            self.log(f"  first sense grounded{(' (heard ' + repr(word) + ')') if word else ''}")
            return root.label
        exp = self._expectation(x)                           # a question it asked predicts this? let the
        if exp:                                              # thought steer perception (top-down)
            base, dev, h = exp
            word = word or h.get("word")                     # a name it HEARD can name what it now sees
            # one observation can answer several questions about the same base at once
            answered = [q for q in self.hypotheses if q["base"] is base and (q["dim"], q["sign"]) in dev]
            for q in answered: self.hypotheses.remove(q)
            self.affect["satisfaction"] += 3.0 * len(answered)
            self.affect["surprise"] = 0.5 * self.affect["surprise"] + 0.5 * base.distance(x)
            child = self._mint(base, x, dev, word)           # question + one observation = a concept
            qs = " / ".join(q["question"] for q in answered)
            self.log(f"  ANSWERED: I wondered '{qs}' -> yes"
                     + (f", it's called '{word}'" if word else "")
                     + f"   -> minted {child.label} from ONE example   (felt: satisfied)")
            self.tokens_spent += 1
            return child.label
        c, dev = self.explain(x)
        surprise = c.distance(x) if dev else 0.0              # a prediction (the sense) failed by this much
        self.affect["surprise"] = 0.5 * self.affect["surprise"] + 0.5 * surprise
        new_word = bool(word) and not any(n.word_counts.get(word, 0) for n in self.nodes)
        if dev and new_word:
            # A WORD IS THE TRIGGER. Hear "bear" for a thing that isn't just a dog, see the difference,
            # and make the node NOW - not after three bears. Someone else already paid the cost of
            # deciding this is a kind; the word carries that evidence. From here on "fire bear" is
            # bear + fire, never big-dog + wide-paws + ... + fire.
            child = self._mint(c, x, dev, word)
            self.log(f"  HEARD '{word}' for something that isn't just {c.label} -> minted {word} from ONE example")
            self.tokens_spent += 1
            return child.label
        if word:
            c.word_counts[word] += 1                         # handle binds by co-occurrence
            c.called.add(word)                               # (a word shared by several kinds becomes a
                                                             #  tree above them - at think()/sleep(), §20)
        if not dev:                                          # fits the sense -> sharpen it
            c.absorb(x); self.tokens_spent += 1
            return c.label
        # doesn't fit: explain as c ∘ descriptors; remember it, WITH its company (candidate for a split)
        c.instances.append((x, dev, word, ctx))
        self.tokens_spent += 1 + len(dev)
        desc = self.describe(x)
        self._maybe_split(c, dev)
        return desc

    # -- the split: recurrence by overlap, decided by compression ---------------------------
    @staticmethod
    def _jaccard(a: tuple, b: tuple) -> float:
        A, B = set(a), set(b)
        return len(A & B) / len(A | B) if (A | B) else 0.0

    def _group(self, c: Concept, bundle: tuple) -> list:
        """Stored instances under c whose bundle overlaps `bundle` enough to be the same kind."""
        # return the ORIGINAL tuple objects so _split can remove them by identity
        return [t for t in c.instances if self._jaccard(t[1], bundle) >= OVERLAP]

    @staticmethod
    def _core(group: list) -> tuple:
        """Descriptors present in at least half the group: what actually recurs (flicker drops out)."""
        cnt = Counter(d for t in group for d in t[1])
        core = [d for d, k in cnt.items() if k * 2 > len(group)]     # strict majority: a 50/50 mix can't define it
        return tuple(sorted(core, key=lambda d: FEATURES.index(d[0])))

    def split_cost(self, c: Concept, bundle: tuple) -> Optional[tuple]:
        group = self._group(c, bundle)
        n = len(group)
        if n < MIN_RECUR:
            return None
        core = self._core(group)
        cost_stay = sum(1 + len(t[1]) for t in group)        # what those instances actually cost
        cost_new = NODE_COST + len(core) + n                  # node + core stored once + 1 token each
        return group, core, cost_stay, cost_new

    def _maybe_split(self, c: Concept, bundle: tuple) -> Optional[Concept]:
        r = self.split_cost(c, bundle)
        if r is None:
            return None
        group, core, cost_stay, cost_new = r
        if cost_stay <= cost_new or not core:
            return None
        return self._split(c, group, core)

    def _split(self, parent: Concept, group: list, core: tuple) -> Concept:
        taken = {id(t) for t in group}
        parent.instances = [t for t in parent.instances if id(t) not in taken]
        X = np.array([t[0] for t in group])
        # the new sense starts at the group's centre with its PARENT's tolerance (a group may be a
        # mixture, and a mixture's own spread would hide that); it grows its own spread from what fits
        child = Concept(self._next, X.mean(0), np.zeros_like(X[0]), n=1, parent=parent, defined_by=core,
                        prior=parent.spread.copy(), born_from=len(group)); self._next += 1
        for (_, _, w, _) in group:
            if w:
                child.word_counts[w] += 1
                if parent.word_counts[w] > 0: parent.word_counts[w] -= 1
        parent.children.append(child); self.nodes.append(child)
        gain = sum(1 + len(t[1]) for t in group) - (NODE_COST + len(core) + len(group))
        self.affect["satisfaction"] += gain                  # it clicked: the world got cheaper to describe
        self.log(f"  SPLIT: [{child.relation()}] recurred x{len(group)} -> new concept {child.label}"
                 f"   (felt: that clicked, {gain} tokens saved)")
        # a concept must actually EXPLAIN its instances: absorb those that fit the new sense, hold the
        # rest under it - they may split again. Splitting is recursive (bird -> made-thing -> ball / club).
        held = []
        for t in group:
            dev = child.deviations(t[0])
            if dev: held.append((t[0], dev, t[2], t[3]))
            else: child.absorb(t[0])
        if child.n > 1: child.n -= 1                         # drop the centre pseudo-sample
        child.instances = held
        for t in list(held):
            if any(u is t for u in child.instances):
                self._maybe_split(child, t[1])
        return child

    # -- unprompted thinking: idle consolidation -------------------------------------------
    def idle(self) -> int:
        """Re-run the compression decision over everything remembered. Nothing prompted this."""
        made = 0
        for c in list(self.nodes):
            while True:
                hit = None
                for t in list(c.instances):
                    if self._maybe_split(c, t[1]):
                        hit = True; made += 1; break
                if not hit:
                    break
        return made

    # =====================================================================================
    # THINKING: running the tree WITHOUT the world (design.md §5)
    # Learning goes world -> tree. Thinking goes tree -> prediction / imagination / question,
    # and the world's answer (surprise) feeds back into learning.
    # =====================================================================================

    # -- descriptor concepts: minted from splits, reusable across branches ------------------
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

    # -- prediction: fill in what you haven't seen from the structure -----------------------
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

    # -- imagination / analogy: compose a concept with descriptors it never co-occurred with ---
    NOVEL = 1.5   # an imagined thing farther than this from every known concept is NEW

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

    # -- curiosity: the gaps in the tree are questions -------------------------------------
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

    # -- feeling: valence computed from the state of compression ---------------------------
    def feel(self) -> tuple[dict, str]:
        """Read the affect. unease = compression debt (unexplained + open questions);
        curiosity = pull of the best open question; surprise/satisfaction are recent pulses."""
        a = self.affect
        unexplained = sum(len(c.instances) for c in self.nodes)
        a["unease"] = unexplained + 0.5 * len(self.hypotheses)
        a["curiosity"] = max((h["pull"] for h in self.hypotheses), default=0.0)
        cands = [("surprised", a["surprise"], f"something didn't fit (surprise {a['surprise']:.1f})"),
                 ("satisfied", a["satisfaction"] / 3, f"things clicked (+{a['satisfaction']:.0f})"),
                 ("uneasy", a["unease"] / 2, f"{unexplained} things I can't explain, {len(self.hypotheses)} open questions"),
                 ("curious", a["curiosity"] / 2, (self.hypotheses[0]["question"] if self.hypotheses else "nothing to ask"))]
        name, _, detail = max(cands, key=lambda t: t[1])
        if max(t[1] for t in cands) < 0.3:
            name, detail = "calm", "everything fits"
        self.last_felt = f"{name}: {detail}"
        return dict(a), self.last_felt

    EXPECT_SLACK = 1.5   # how far a thought may pull perception toward its own answer

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

    def _link_compound(self, c: Concept, word: str):
        """A compound of known CONCEPT handles wires the web: 'whale shark' is-a shark (the head) and
        is linked to whale (the modifier) - connected to whales despite the differences. 'shark teeth'
        would link teeth to shark the same way."""
        parts = [p for p in word.replace("-", " ").split() if p]
        if len(parts) < 2:
            return
        head = self.concept_for_word(parts[-1])
        if head is not None and head is not c and not self._is_ancestor(head, c):
            c.links.append(("is-a", head)); head.links.append(("has-kind", c))
        for p in parts[:-1]:
            m = self.concept_for_word(p)
            if m is not None and m is not c:
                c.links.append(("assoc", m)); m.links.append(("assoc", c))
                self.log(f"  LINKED: '{word}' <-> {m.label}  (named after it; connected despite the differences)")

    @staticmethod
    def _is_ancestor(a: Concept, c: Concept) -> bool:
        while c is not None:
            if c is a: return True
            c = c.parent
        return False

    # -- the web, read out from a handle ---------------------------------------------------
    def web(self, word: str) -> dict:
        """Everything a handle connects to: 'shark' -> its kinds, what it is (up the tree AND across
        memberships), what it's linked to, and every compound that mentions it."""
        c = self.concept_for_word(word)
        if c is None:
            return {}
        up = []
        p = c.parent
        while p is not None:
            up.append(p.label); p = p.parent
        return {
            "is": up + [g.label for r, g in c.links if r == "is-a"],
            "kinds": [k.label for k in c.children] + [k.label for r, k in c.links if r == "has-kind"],
            "linked": [m.label for r, m in c.links if r == "assoc"],
            "mentioned_in": [n.label for n in self.nodes
                             if n is not c and word in n.label.replace("-", " ").split()],
        }

    # -- hearing a compound word: handles compose the way concepts do ----------------------
    def hear(self, compound: str) -> Optional[dict]:
        """'broadsword' = 'broad' + 'sword'. If both handles are already known, compose the concepts the
        same way -> an expectation of what it will look like BEFORE one is ever seen. One picture then
        confirms it. (The word did half the learning.)"""
        # two CONCEPT handles ('whale shark'): expect a kind of the head, linked to the modifier
        parts = [p for p in compound.replace("-", " ").split() if p]
        if len(parts) >= 2:
            head = self.concept_for_word(parts[-1]); mods = [self.concept_for_word(p) for p in parts[:-1]]
            if head is not None and all(m is not None for m in mods):
                h = {"question": f"'{compound}' = {' + '.join(parts)}", "base": head, "descriptor": None,
                     "dim": None, "sign": 0, "proto": head.proto.copy(), "novelty": 0.0, "pull": 10.0,
                     "word": compound}
                self.hypotheses.insert(0, h)
                self.log(f"  HEARD '{compound}': I know {' and '.join(parts)} -> I expect a kind of "
                         f"{head.label} that has to do with {', '.join(m.label for m in mods)}")
                return h
        D = self.descriptor_concepts()
        for name, d in sorted(D.items(), key=lambda kv: -len(kv[0])):
            if compound.startswith(name) and len(compound) > len(name):
                rest = compound[len(name):].lstrip("-_ ")
                base = self.concept_for_word(rest)
                if base is None:
                    continue
                p, near, dist = self.imagine(base, [name])
                h = {"question": f"'{compound}' = {name} + {rest}", "base": base, "descriptor": name,
                     "dim": d["dim"], "sign": d["sign"], "proto": p, "novelty": dist, "pull": dist + 10.0,
                     "word": compound}
                self.hypotheses.insert(0, h)                   # it now EXPECTS this thing
                self.log(f"  HEARD '{compound}': I know '{name}' and '{rest}' -> I expect a {name} {rest}")
                return h
        return None

    def think(self, top: int = 6) -> dict:
        """One idle 'thought'. Nothing asked for it. Feeling decides what to do with the time:
        uneasy -> consolidate what's unexplained; otherwise -> wonder (curiosity). Then the pulses fade."""
        _, mood_before = self.feel()
        made = self._generalize_all()                         # the day's words reorganise the tree
        if self.affect["unease"] >= 1.0:
            made += self.idle()                               # settle the debt
        qs = self.wonder(top)
        returned = self._return_to_wants()                    # then: back to what it wants
        qs = self.hypotheses[:top]
        self.affect["surprise"] *= 0.5; self.affect["satisfaction"] *= 0.5     # pulses fade
        _, mood_after = self.feel()
        return {"mood_before": mood_before, "consolidated": made, "descriptors": self.descriptor_concepts(),
                "questions": qs, "returned_to": returned, "mood_after": mood_after}

    # -- handles: many-to-many. A word is a door into a SET of nodes (doubles) --------------
    def concepts_for_word(self, word: str) -> list[Concept]:
        """Every node this handle opens. More than one = a DOUBLE (to / too / two)."""
        return [c for c in self.nodes if c.word_counts.get(word, 0) >= MIN_RECUR]

    def resolve(self, word: str) -> tuple[Optional[Concept], list[Concept]]:
        """Pick which sense of a handle is meant, from CONTEXT: the candidate that most often lights
        up together with what is active right now. Returns (choice or None if truly ambiguous,
        all candidates)."""
        cands = self.concepts_for_word(word)
        if len(cands) <= 1:
            return (cands[0] if cands else None), cands
        active = self.active_nodes()
        if not active:
            return None, cands
        # company, re-read through the CURRENT tree: for each candidate, how often did one of ITS
        # observations arrive right after one of the kinds that are active now? (smoothed one step
        # up the tree). What's already in mind wins outright - it is the most expected thing of all.
        who = [self.nearest(x) for x in self.obs]
        def near_active(c):
            return c in active or c.parent in active or any(a.parent is c for a in active)
        scored = []
        for c in cands:
            s = 1000.0 if c in active else 0.0
            for i, w in enumerate(who):
                if w is c:
                    for j in range(max(0, i - self.WINDOW), i):
                        if who[j] is c: continue
                        if who[j] in active: s += 1.0
                        elif near_active(who[j]): s += 0.5
            scored.append((s, c))
        scored.sort(key=lambda t: -t[0])
        if scored[0][0] == 0 or (len(scored) > 1 and scored[0][0] == scored[1][0]):
            return None, cands                               # nothing in mind tips it: ambiguous
        return scored[0][1], cands

    # -- recall: the past, re-read through the present -------------------------------------
    def recall(self, word: str) -> list[tuple[int, str, str]]:
        """'Have you seen a bat before?' Memory stores experiences, not labels. Every observation is
        re-read through the CURRENT tree; the ones that resolve to this word's concept(s) are the
        answer - including ones it called something else at the time. Returns
        (observation index, what it called it then, what it reads it as now)."""
        cands = self.concepts_for_word(word)
        if not cands:
            return []
        hits = []
        for i, x in enumerate(self.obs):
            now = self.nearest(x)
            if now in cands:
                hits.append((i, self.said[i] if i < len(self.said) else "?", now.label))
        # remembering something keeps it - but recalling a crowd doesn't fix every face:
        # the boost is shared across the hits (THE lion gets +1; one dog of twenty gets +0.05)
        for i, _, _ in hits:
            self.weight[i] += 1.0 / len(hits)
        return hits

    # -- forgetting: priority, not time ----------------------------------------------------
    DECAY, FORGET = 0.5, 0.75

    def centrality(self, c: Concept) -> float:
        """How much a concept matters structurally: what hangs off it."""
        return (len(c.children) + len(c.defined_by)) / 4.0

    def sleep(self) -> dict:
        """A night. Raw memories decay by priority; the ones that mattered (surprising, significant,
        re-read, or tied to a central concept) survive, the rest are let go. Concepts are untouched -
        the event goes, the reading stays. Open questions are NOT kept: a question that still matters
        is regenerated from the tree's gaps the next time it idles."""
        before = len(self.obs)
        keep = []
        self._generalize_all()                               # reorganise before forgetting
        self._refresh_wants()
        wanted = {id(w.concept): abs(w.valence) for w in self.wants}
        who = [self.nearest(x) for x in self.obs]
        for i, x in enumerate(self.obs):
            self.weight[i] *= self.DECAY
            # an EVENT survives on its own priority: how surprising it was, whether it made a concept,
            # how often it's been revisited. Belonging to a central concept does NOT save an event -
            # you forget the washer even though work is central. The concept is what centrality keeps.
            # But an event tied to something it WANTS gets the goal term: that's "matters to me".
            pri = self.weight[i] + wanted.get(id(who[i]), 0.0)
            if pri >= self.FORGET:
                keep.append(i)
            elif self.reward[i]:
                # the event goes, the FEELING stays: fold it into the concept it reads as today
                who[i].v_sum += self.reward[i]; who[i].v_n += 1
                who[i].v_watched += (self.src[i] == "watched")
        self.obs = [self.obs[i] for i in keep]
        self.said = [self.said[i] for i in keep]
        self.weight = [self.weight[i] for i in keep]
        self.reward = [self.reward[i] for i in keep]
        self.src = [self.src[i] for i in keep]
        self.recent = []
        dropped_q = len(self.hypotheses)
        self.hypotheses = []                                 # questions regenerate; wants do NOT get dropped
        self.forgotten += before - len(keep)
        self.log(f"  SLEEP: let go of {before - len(keep)} of {before} raw memories and {dropped_q} open "
                 f"questions; kept {len(keep)} that mattered. Every concept intact; "
                 f"{len(self.wants)} want(s) carried over.")
        return {"forgot": before - len(keep), "kept": len(keep), "questions_dropped": dropped_q}

    # -- RETROACTIVELY ADDING TREES: a generic word inserts a parent above existing kinds ----------
    def _maybe_generalize(self, word: str) -> Optional[Concept]:
        """'Dogs and cats are both mammals, and animals; fish are animals without being mammals.'
        When one word is bound to several existing concepts, it may name what they SHARE: build the
        abstract node, insert it above them, re-parent them as siblings, re-express each relative to it.
        The historical chain (cat under dog) dissolves the moment a word gives it a reason to.
        Compression tells a GENERIC from a DOUBLE: 'mammal' over dog+cat costs little; 'bat' over the
        animal and the club would cost everything, so that stays two meanings."""
        members = [n for n in self.nodes if n.word_counts.get(word, 0) >= MIN_RECUR]
        if len(members) < 2:
            return None
        # a word that is some concrete thing's OWN NAME is a name, not a generic: 'bat' names the animal
        # and names the club - that's a DOUBLE, resolved by context, never a tree. A generic (mammal,
        # animal, pet) is a second name laid over things that already have their own.
        if any(m.n > 1 and m.label == word for m in members):
            return None
        has = lambda lst, c: any(m is c for m in lst)
        provisional = [m for m in members if m.n <= 1 and m.label == word]
        # collapse upward: if every child of some node is a member, the node stands in for them
        # ('animal' on dog, cat, fish -> dog and cat are all of mammal's children -> {mammal, fish})
        changed = True
        while changed:
            changed = False
            for P in self.nodes:
                if has(members, P) or not P.children or P.n > 1:
                    continue                                 # only a HOLLOW node stands in for its children;
                # ...and only over the children that BELONG there by word. A child filed under P by
                # nearest-neighbour accident (fish under mammal, never CALLED a mammal) neither counts
                # toward the collapse nor blocks it - it stays a member on its own, and the generic goes
                # above P with the accident pulled out. An unnamed hollow has nothing to be called.
                legit = (list(P.children) if P.label.startswith("concept#")
                         else [k for k in P.children if P.label in k.called])
                if legit and all(has(members, k) for k in legit):
                    members = [m for m in members if not has(legit, m)] + [P]
                    changed = True
                    break
        if len(members) == 1:                                # the word names an EXISTING node - a hollow
            G = members[0]                                   # abstraction finally gets its name
            G.word_counts[word] = max(G.word_counts.get(word, 0), MIN_RECUR); G.called.add(word)
            for n in self.nodes:
                if n is not G and word in n.word_counts: del n.word_counts[word]
                if self._is_ancestor(G, n): n.called.add(word)
            for p in provisional:                             # a stand-in node this word made earlier
                if p is not G: self._dissolve(p)             # gives way to the real one
            self.log(f"  NAMED: '{word}' is what everything under {G.label} shares -> that node is '{word}'")
            return G
        # words arrive one member at a time: if THE generic for this word already exists - a hollow node
        # whose own identity is this word (not a provisional alias) - newcomers JOIN it
        existing = [m for m in members if m.n <= 1 and m.label == word]
        if len(existing) == 1:
            G = existing[0]
            for o in members:
                if o is G: continue
                if word in o.word_counts: del o.word_counts[word]
                o.called.add(word)
                if o.parent is G:
                    continue                                 # already a direct child: nothing to do
                if self._is_ancestor(G, o):                      # already under it, via the chain
                    if len(G.deviations(o.proto)) < len(o.defined_by):    # strictly shorter as a direct child?
                        self._reparent(o, G)
                        self.log(f"  JOINED: {o.label} is a {G.label} directly -> {o.label} = {o.relation()}")
                    continue
                if not self._would_strip(o, G):
                    self._reparent(o, G)
                    self.log(f"  JOINED: {o.label} is also a {G.label} -> {o.label} = {o.relation()}")
                else:
                    o.links.append(("is-a", G)); G.links.append(("has-kind", o))
                    self.log(f"  JOINED: {o.label} is also a {G.label} (membership - it belongs elsewhere too)")
            return G
        # where would a parent go? above the topmost member. Pulling a member up there is fine when it
        # only leaves behind concrete kinds and abstractions it was never called ('fish' under 'dog'
        # under 'mammal' is the accident - fish was never called a mammal). It is NOT fine when it would
        # strip an abstraction the member actually WAS called ('pet' pulling dog out of mammal) -> cross-cut.
        def depth(c): return 0 if c.parent is None else 1 + depth(c.parent)
        top = min(members, key=depth)
        P = top.parent
        if any(self._would_strip(m, P) for m in members):
            return self._cross_cut(word, members, "it cuts across branches")
        # the abstract node: the members' common centre, with a member-sized tolerance. A generic is
        # about KINDS, not instances: a child who has seen a hundred dogs and one fish still puts
        # "animal" in the middle. Each member counts once.
        protos = np.array([m.proto for m in members])
        centre = protos.mean(0)
        prior = np.max([m.spread for m in members], axis=0)
        G = Concept(self._next, centre, np.zeros_like(centre), n=1, parent=P, prior=prior,
                    born_from=len(members)); self._next += 1
        # the WORD has the authority here ("I retroactively add trees"): a locally-cheap accident
        # (shark filed under goldfish) is never undone by compression alone. Reported, not gated.
        old = sum(len(m.defined_by) for m in members)
        new = sum(len(G.deviations(m.proto)) for m in members) + (len(P.deviations(centre)) if P else 0)
        # insert G above the members
        G.defined_by = P.deviations(centre) if P else ()
        if P:
            P.children = [k for k in P.children if not has(members, k)] + [G]
        for m in members:
            if m.parent:                                     # one parent each: always detach from the old one
                m.parent.children = [k for k in m.parent.children if k is not m]
            m.parent = G; G.children.append(m)
            m.defined_by = G.deviations(m.proto)
            m.called.add(word)                               # it was (through its leaves) called this
        for n in self.nodes:                                 # the generic takes the word; the leaves keep
            if word in n.word_counts: del n.word_counts[word]  # only the memory of having been called it
        G.word_counts[word] = MIN_RECUR; G.called.add(word)
        self.nodes.append(G)
        for p in provisional:
            if p is not G: self._dissolve(p)
        rel = "; ".join(f"{m.label} = {m.relation()}" for m in members)
        self.log(f"  GENERALIZED: '{word}' = what {' and '.join(m.label for m in members)} share -> "
                 f"inserted above them ({old} -> {new} descriptors). {rel}")
        return G

    def _dissolve(self, p: Concept):
        """Remove a hollow node: its children go to its parent."""
        for k in list(p.children):
            self._reparent(k, p.parent) if p.parent else setattr(k, "parent", None)
        if p.parent: p.parent.children = [k for k in p.parent.children if k is not p]
        self.nodes = [n for n in self.nodes if n is not p]

    def _generalize_all(self) -> int:
        """Consolidation: the day's generic words reorganise the tree, SPECIFIC FIRST (mammal before
        animal, so animal can collapse over mammal). This happens when it idles or sleeps, not
        mid-sentence - restructuring on the second utterance built nodes everything later collided with."""
        made = 0
        for _ in range(4):
            words = Counter()
            for n in self.nodes:
                for w, k in n.word_counts.items():
                    if k >= MIN_RECUR: words[w] += 1
            cands = sorted([w for w, k in words.items() if k >= 2], key=lambda w: words[w])
            changed = False
            for w in cands:
                if self._maybe_generalize(w) is not None:
                    made += 1; changed = True
            if not changed: break
        return made

    def _reparent(self, o: Concept, G: Concept):
        if o.parent: o.parent.children = [k for k in o.parent.children if k is not o]
        o.parent = G; G.children.append(o); o.defined_by = G.deviations(o.proto)

    def _ancestors(self, c: Concept) -> list:
        out = []; p = c.parent
        while p is not None: out.append(p); p = p.parent
        return out

    def _would_strip(self, m: Concept, new_parent: Optional[Concept]) -> bool:
        """Would moving m under new_parent take away an abstraction m was actually CALLED?"""
        keep = {id(a) for a in ([new_parent] + self._ancestors(new_parent))} if new_parent else set()
        return any(id(a) not in keep and a.n <= 1 and a.label in m.called for a in self._ancestors(m))

    def _cross_cut(self, word: str, members: list, why: str) -> Concept:
        """A generic that can't be a parent becomes a MEMBERSHIP across branches: 'pet' on dog, cat,
        goldfish; 'fish' on goldfish and shark. A hub in the web, with no single place in the tree."""
        protos = np.array([m.proto for m in members])
        G = Concept(self._next, protos.mean(0), np.zeros_like(protos[0]), n=1, parent=None,
                    prior=np.max([m.spread for m in members], axis=0), born_from=len(members))
        self._next += 1
        G.word_counts[word] = MIN_RECUR; G.called.add(word)
        for m in members:
            m.links.append(("is-a", G)); G.links.append(("has-kind", m)); m.called.add(word)
        for n in self.nodes:
            if word in n.word_counts: del n.word_counts[word]
        self.nodes.append(G)
        self.log(f"  CROSS-CUT: '{word}' = {', '.join(m.label for m in members)} - a membership, not a parent ({why})")
        return G

    # =====================================================================================
    # ANALOGY: a handle is a library (design.md §21)
    # =====================================================================================
    def define(self, word: str, parts: list) -> Struct:
        """Bind a word to a structure of other concepts. Parts that don't exist yet become bare ideas."""
        for p in parts:
            if self.concept_for_word(p) is None and p not in self.structs:
                self.structs[p] = Struct(p, [])
        old = self.structs.get(word)
        s = Struct(word, list(parts)); self.structs[word] = s
        s.links = [("has-part", p) for p in parts]
        if old is not None:                                  # told what it's made of: keep what it IS
            s.links += [l for l in old.links if l[0] != "has-part"]
        self.log(f"  DEFINED '{word}' = {' + '.join(parts)}")
        return s

    def _ensure(self, name: str):
        if self.concept_for_word(name) is None and name not in self.structs:
            self.structs[name] = Struct(name, [])

    def frame(self, word: str, roles: dict, story: str = "") -> Struct:
        """A structure WITH its relations. The sum is the handle; the roles are the parts. If its roles
        contain another frame's, it SPECIALISES it (bankai is a kind of transforming weapon)."""
        s = Struct(word, [f for f in roles.values() if f], roles=dict(roles), story=story, is_frame=True)
        old = self.structs.get(word)
        if old is not None:                                  # re-induced: keep its instances and lineage
            s.links += [l for l in old.links if l[0] in ("has-kind", "progression-of")]
        self.structs[word] = s
        for r, f in roles.items():
            s.links.append(("role", r))
            if f: self._ensure(f); self.role_fills[(r, f)] += 1
        for other in list(self.structs.values()):            # only a FRAME can be specialised
            if other is not s and other.is_frame and other.roles and set(other.roles) < set(roles):
                s.links.append(("is-a", other.name)); other.links.append(("has-kind", word))
                self.log(f"  '{word}' specialises '{other.name}'  (adds: {', '.join(sorted(set(roles) - set(other.roles)))})")
        self.log(f"  FRAME '{word}' = " + ", ".join(f"{r}={f or '?'}" for r, f in roles.items()))
        return s

    # -- a frame from EXPERIENCE: a recurring sequence whose fillers vary ------------------------
    def episode(self, name: str, steps: list, labels: Optional[list] = None, who: Optional[str] = None) -> Optional[Struct]:
        """One experience of something happening, in order. 'Every zanpakuto goes through basic sword >
        command > transform > new sword when shikai is used.' After two or more episodes of the same
        thing, the positions where the fillers VARY become roles (slots), the positions that stay the
        same become fixed parts, and the narration's labels name the positions.
        With `who`, the episode is also KEPT as a named instance ("ichigo's bankai") that is-a the frame
        - the before, the after, AND the becoming, each its own thing - so you can describe Ichigo's
        bankai, and his shikai, while separately talking about bankai as a whole."""
        self.episodes.setdefault(name, []).append(list(steps))
        if labels: self.episode_labels[name] = self._unique(list(labels))
        for s in steps: self._ensure(s)
        if who and labels:
            labels = self.episode_labels[name]
            inst = f"{who}'s {name}"
            s = Struct(inst, list(steps), roles=dict(zip(labels, steps))); self.structs[inst] = s
            s.links.append(("is-a", name))
            self.structs.setdefault(name, Struct(name, [])).links.append(("has-kind", inst))
            for i, lab in enumerate(labels):                 # a change = before, BY, becoming, after
                if self._is_change(lab) and 0 < i < len(steps) - 1:
                    is_state = lambda j: not self._is_change(labels[j]) and "command" not in labels[j]
                    before = next((steps[j] for j in range(i - 1, -1, -1) if is_state(j)), steps[i - 1])
                    after = next((steps[j] for j in range(i + 1, len(steps)) if is_state(j)), steps[i + 1])
                    by = next((steps[j] for j in range(i - 1, -1, -1) if "command" in labels[j]), None)
                    roles = {"before": before, **({"by": by} if by else {}), "becoming": steps[i], "after": after}
                    c = Struct(f"{who}'s {lab}", list(roles.values()), roles=roles)
                    self.structs[c.name] = c
                    c.links.append(("transition-of", inst)); s.links.append(("has-change", c.name))
                    for ro, f in c.roles.items(): self.role_fills[(ro, f)] += 1
        eps = self.episodes[name]
        if len(eps) >= 2 and all(len(e) == len(eps[0]) for e in eps):
            return self._induce(name)
        return None

    @staticmethod
    def _unique(labels: list) -> list:
        seen, out = Counter(), []
        for l in labels:
            seen[l] += 1
            out.append(l if seen[l] == 1 else f"{l}-{seen[l]}")
        return out

    def _is_change(self, label: str) -> bool:
        base = label.split("-")[0]
        return any(k in label for k in ("transform", "change", "becom")) or base in self._alike("transformation")

    def _induce(self, name: str) -> Struct:
        eps = self.episodes[name]; L = len(eps[0])
        labels = self.episode_labels.get(name) or [f"step-{i + 1}" for i in range(L)]
        roles = {}
        for i in range(L):
            fillers = [e[i] for e in eps]
            if len(set(fillers)) > 1:
                roles[labels[i]] = None                      # a slot: whatever varies here
                for f in fillers:                            # ...and every filler seen there is a connection
                    self.role_fills[(labels[i], f)] += 1
            else:
                roles[labels[i]] = fillers[0]                # a fixed part
        s = self.frame(name, roles, story="every time: " + " > ".join(labels))
        # a PROGRESSION: this sequence begins with another one's and goes on (bankai = shikai, then more)
        for other, oeps in self.episodes.items():
            if other == name or not oeps: continue
            olab = self.episode_labels.get(other) or [f"step-{i + 1}" for i in range(len(oeps[0]))]
            if len(olab) < L and labels[:len(olab)] == olab:
                if not any(r == "progression-of" and g == other for r, g in s.links):
                    s.links.append(("progression-of", other))
                    self.log(f"  '{name}' is a progression of '{other}': the same steps, then more")
        return s

    def equate(self, a: str, b: str):
        """A simple connection: transform = change. Roles and words that mean the same thing."""
        self.same.setdefault(a, set()).add(b); self.same.setdefault(b, set()).add(a)

    def _alike(self, r: str) -> set:
        return {r} | self.same.get(r, set())

    def fit(self, filler: str, role: str) -> tuple:
        """How well a filler suits a role - GRADED, from accumulated connections, never a type check.
        A button CAN be a weapon (throw it; press it to detonate a bomb); it's just more likely a
        trigger. Returns (likelihood, why)."""
        roles = self._alike(role)
        n = sum(self.role_fills[(r, filler)] for r in roles)
        total = sum(k for (r, f), k in self.role_fills.items() if r in roles)
        if n:
            return min(1.0, 0.5 + 0.5 * n / max(1, total)), f"seen as the {role} {n}x"
        for g in self._generic_chain(filler):
            if any(self.role_fills[(r, g)] for r in roles):
                return 0.6, f"a {g} has been the {role}"
        # the other way up: kinds OF this filler have played the role (three sealed swords were the weapon)
        kinds = [f for (r, f), k in self.role_fills.items() if r in roles and k
                 and filler in self._generic_chain(f)]
        if kinds:
            return 0.6, f"kinds of {filler} have been the {role} ({', '.join(sorted(set(kinds))[:3])})"
        if filler in self.structs or self.concept_for_word(filler) is not None:
            return 0.1, "possible - nothing says it can't"
        return 0.0, "unknown"

    def recognize(self, fillers: dict, top: int = 3) -> list:
        """Role-filling, not overlap - and GRADED: which frames do these fillers fit, and how well?
        {weapon: sword, command: button, ...} -> bankai, likely. The same parts with the roles
        scrambled -> bankai, possible but unlikely. Returns (likelihood, frame, missing roles, fits)."""
        out = []
        for s in self.structs.values():
            if not s.roles or not s.is_frame: continue       # you recognise a PATTERN, not someone's instance
            filled = [r for r in s.roles if r in fillers]
            if not filled: continue
            fits = {r: self.fit(fillers[r], r) for r in filled}
            coverage = len(filled) / len(s.roles)
            likelihood = coverage * (sum(v for v, _ in fits.values()) / len(fits))
            out.append((coverage, likelihood, len(s.roles), s.name, [r for r in s.roles if r not in fillers], fits))
        # SHAPE first (do the roles get filled?), then EXPERIENCE (has anything like this filled them?)
        out.sort(key=lambda t: (-t[0], -t[1], -t[2]))
        return [(c, l, n, miss, fits) for c, l, _, n, miss, fits in out[:top]]

    # -- INTUITION: a word it was never told, placed by how it is used (design.md §26) ------------
    def _cues(self, word: str) -> set:
        """Everything that has happened around a word: the frame roles it has filled (zanpakuto was
        the tool of a cut), the frames it CAN do, its parts, what it was likened to."""
        cues = set()
        for s in self.structs.values():
            if s.is_frame or not s.roles: continue
            frame = next((g for r, g in s.links if r == "is-a"), None)
            for ro, f in s.roles.items():
                if f == word and frame: cues.add(("role", frame, ro))
        s = self.structs.get(word)
        if s is not None:
            for r, g in s.links:
                if r == "can": cues.add(("role", g, "*"))
                elif r in ("has-part", "like"): cues.add((r, g))
        return cues

    def _kinds_of(self, name: str, depth: int = 3) -> list:
        out, k = [], name
        while k in self.structs and len(out) < depth:
            k = next((g for r, g in self.structs[k].links if r == "is-a"), None)
            if k is None or k in out or k == name: break
            out.append(k)
        return out

    def intuit(self, word: str, top: int = 3) -> list:
        """Never told what a zanpakuto is - but it cut things, like a knife did; it has a handle, like a
        sword; it was swung like a whip. Score every known thing by the cues it shares, pass each
        score up to that thing's kinds (sword -> weapon -> tool), and the heaviest is the guess.
        Returns [(kind, score, because)] - the guess and the evidence, so it can be told it's wrong."""
        mine = self._cues(word)
        if not mine: return []
        others = {k: self._cues(k) for k, s in self.structs.items() if k != word and not s.is_frame and not s.roles}
        # RARITY: a cue nearly everything has ("a") says almost nothing; one that two things share says a lot.
        rare = lambda n: 1.0 / math.log2(1 + max(n, 1))
        # "knife can cut" and "the cook cut the bread with a knife" are the same cue: the knife does cutting
        has = lambda th, c: c in th or (c[0] == "role" and ("role", c[1], "*") in th)
        near = lambda th, c: c[0] == "role" and any(t[0] == "role" and t[1] == c[1] for t in th)
        kin = lambda c, k: c[0] == "like" and (c[1] == k or self._kinds_of(c[1])[:1] == self._kinds_of(k)[:1] != [])
        n_has = {c: sum(has(t, c) for t in others.values()) for c in mine}
        n_near = {c: sum(near(t, c) for t in others.values()) for c in mine}
        n_kin = {c: sum(kin(c, k) for k in others) for c in mine}
        # NEGATIVE cues are concepts too: "can't smash" = can no-smash, and no-smash is the OPPOSITE of
        # smash. Sharing no-smash counts like any cue; doing its opposite is a flat penalty (can't 1.0,
        # hard 0.6) - ruling something out doesn't weaken because many things could do it. And it is
        # the same opposite link hot/cold use: it's very hard to be hot and cold, blunt and sharp.
        def against(th, act):
            for rel, opp in self.structs.get(act, Struct(act, [])).links:
                w = {"opposite": 1.0, "partly-opposite": 0.6}.get(rel)
                if w and near(th, ("role", opp, "*")): return w, opp
            return None
        def opposites(act):
            return [o for rel, o in self.structs.get(act, Struct(act, [])).links if rel == "opposite"]
        def broken(cues, holder):
            """ONLY: ice can only be cold. Anything warm is not ice - not 'less likely', not ice."""
            onlys = [g for r, g in self.structs.get(holder, Struct(holder, [])).links if r == "only"]
            return any(near(cues, ("role", o, "*")) for y in onlys for o in opposites(y))
        score, why = Counter(), {}
        for k, theirs in others.items():
            if broken(mine, k) or broken(theirs, word): continue
            shared = []
            for c in mine:
                if has(theirs, c): shared.append((rare(n_has[c]), c))
                elif near(theirs, c): shared.append((0.5 * rare(n_near[c]), ("near",) + c[1:]))
                elif kin(c, k): shared.append((rare(n_kin[c]), c))
                elif c[0] == "role" and against(theirs, c[1]):
                    w, opp = against(theirs, c[1]); shared.append((-w, ("unlike", c[1], opp)))
            if not shared or all(w < 0 for w, _ in shared): continue
            score[k] = sum(w for w, _ in shared)
            why[k] = sorted(f"{k}: {self._say_cue(c)}" for _, c in shared)
        return [(k, round(v, 2), why[k]) for k, v in score.most_common(top)]

    def guess(self, word: str, floor: float, margin: float):
        """The most SPECIFIC kind the evidence settles on. One clear winner -> a kind of that. Several
        tied (knife, sword, axe) -> what they have in common (tool): it knows the shape without the
        species. Tied with nothing in common -> no guess. Returns (kind, score, why, tied) or None."""
        res = self.intuit(word, top=6)
        if not res or res[0][1] < floor: return None
        tied = [r for r in res if res[0][1] - r[1] < margin]
        if len(tied) == 1:
            k, sc, why = tied[0]; return k, sc, why, [k]
        common = set(self._kinds_of(tied[0][0])[:1])
        for k, _, _ in tied[1:]: common &= set(self._kinds_of(k)[:1])
        if not common: return None
        return common.pop(), tied[0][1], [w for _, _, ws in tied for w in ws[:1]], [k for k, _, _ in tied]

    def _say_cue(self, c) -> str:
        if c[0] == "role": return f"it was the {c[2]} of a {c[1]}" if c[2] != "*" else f"it can {c[1]}"
        if c[0] == "near": return f"it was in a {c[1]} too"
        if c[0] == "unlike": return f"this can {c[1]}, but that can {c[2]}"
        if c[0] == "has-part": return f"it has a {c[1]}"
        return f"it is like a {c[1]}"

    def instantiate(self, word: str, frame: str, fillers: dict) -> Struct:
        """It's a bankai AND its own thing: a node of its own, is-a the frame, roles filled."""
        s = Struct(word, list(fillers.values()), roles=dict(fillers)); self.structs[word] = s
        s.links.append(("is-a", frame)); self.structs[frame].links.append(("has-kind", word))
        for ro, f in fillers.items():
            self._ensure(f); self.role_fills[(ro, f)] += 1      # one more simple connection
        self.log(f"  '{word}' is a {frame}:  " + ", ".join(f"{r}={f}" for r, f in fillers.items()))
        return s

    def tell(self, word: str, listener: set, depth: int = 0) -> str:
        """The shortest description THIS listener can decompress. A handle is only a compression if
        the receiver holds it: 'Bankai in real life' to someone who watched Bleach; 'a sword that can
        become gauntlets' to someone who didn't. Falls back to the most specific shared generic, then
        to the roles, spelled out in words the listener has."""
        if word in listener or depth > 3:
            return word
        s = self.structs.get(word)
        if s is None:
            return word
        # 1. the most specific generic the listener already knows, anywhere up the is-a chain, plus
        #    only what's specific about THIS one ("a transforming-weapon: button as the command")
        for g in self._generic_chain(word):
            if g in listener:
                groles = self.structs[g].roles if g in self.structs else {}
                extra = [(ro, f) for ro, f in s.roles.items() if f and ro not in groles]
                return f"a {g}" + (": " + ", ".join(f"{self.tell(f, listener, depth + 1)} as the {ro}"
                                                    for ro, f in extra) if extra else "")
        # 2. its own filled roles, in words the listener has ("a sword that can become spiked-gauntlet")
        fillers = {ro: self.tell(f, listener, depth + 1) for ro, f in s.roles.items() if f}
        if fillers:
            if {"weapon", "new-form"} <= set(fillers):
                return f"a {fillers['weapon']} that can become {fillers['new-form']}"
            return ", ".join(f"{f} as the {ro}" for ro, f in fillers.items())
        # 3. an abstract frame with nothing filled: go up, but never accept an empty rendering
        for g in self._generic_chain(word):
            via = self.tell(g, listener, depth + 1)
            if via and via != g:
                return via
        if s.parts:
            return " + ".join(self.tell(p, listener, depth + 1) for p in s.parts)
        if s.story:                                          # last resort: the story it came with
            return s.story.replace("every time: ", "")
        return word

    def _generic_chain(self, word: str) -> list:
        """Everything this is a kind of, most specific first (bankai, transforming-weapon, ...)."""
        out, seen, frontier = [], {word}, [word]
        while frontier:
            nxt = []
            for w in frontier:
                for r, g in self.structs.get(w, Struct(w, [])).links:
                    if r == "is-a" and g not in seen:
                        seen.add(g); out.append(g); nxt.append(g)
            frontier = nxt
        return out

    def breakdown(self, word: str) -> str:
        """Break it down for someone who has never heard it: the story, the parts with their roles,
        and the general term. (Not `explain` - that name is the perceptual path's and I shadowed it.)"""
        s = self.structs[word]
        lines = [f"{word}:"] + ([f"  {s.story}"] if s.story else [])
        lines += [f"  - {r}: {f or '(anything that fits)'}" for r, f in s.roles.items()]
        gen = [o for r, o in s.links if r == "is-a"]
        if gen: lines.append(f"  generally: a {gen[0]}")
        return "\n".join(lines)

    def structure_of(self, name: str) -> set:
        """What a thing is MADE OF, as a set of names: its structural parts (and anything it borrowed),
        plus - for a perceptual concept - the descriptors that define it and what it's linked to."""
        parts = set()
        if name in self.structs:
            s = self.structs[name]
            parts |= set(s.parts) | {p for p, _ in s.borrowed} | set(s.roles)
        c = self.concept_for_word(name)
        if c is not None:
            if c.parent:
                parts |= {dev_word(d, c.proto[FEATURES.index(d)], c.parent.proto[FEATURES.index(d)])
                          for d, _ in c.defined_by}
            parts |= {m.label for r, m in c.links if r in ("assoc", "is-a")}
        return parts

    def like(self, name: str, top: int = 3) -> list:
        """'Oh, like Byakuya's bankai.' The things whose STRUCTURE overlaps this one's, wherever they sit
        in the tree - surface be damned. Returns (overlap, other, shared parts, parts it could borrow)."""
        mine = self.structure_of(name)
        out = []
        for other in list(self.structs) + [c.label for c in self.nodes]:
            if other == name: continue
            theirs = self.structure_of(other)
            shared = mine & theirs
            if shared:
                out.append((len(shared) / len(mine | theirs), other, sorted(shared), sorted(theirs - mine)))
        out.sort(key=lambda t: -t[0])
        return out[:top]

    def borrow(self, target: str, source: str, part: str) -> Struct:
        """'Yes, exactly like Senbonzakura' - or just its safe zone. Take ONE addressable part of a
        structure and give it to something else, however unrelated. A robot gets a region it may only
        operate in if its owner permits, from a sword."""
        if part not in self.structure_of(source):
            raise ValueError(f"{source} has no part '{part}'")
        t = self.structs.setdefault(target, Struct(target, []))
        t.borrowed.append((part, source))
        t.links.append(("has-part", part)); t.links.append(("like", source))
        self.log(f"  BORROWED: {target} takes '{part}' from {source}"
                 + (f"  ({part} = {' + '.join(self.structs[part].parts)})" if part in self.structs and self.structs[part].parts else ""))
        return t

    # -- the old link, kept: history you can use -------------------------------------------
    def used_to_think(self, c: Concept) -> str:
        """The node replaced a reading, but the reading isn't gone. 'I used to think a bear was a
        big, wide-pawed dog.' From its definition, and from what it actually SAID before the node existed."""
        if not c.parent:
            return f"{c.label} is as far back as I go"
        old = [self.said[i] for i, x in enumerate(self.obs)
               if i < len(self.said) and self.nearest(x) is c and self.said[i] != c.label]
        line = f"I used to think a {c.label} was a {c.relation()}"
        if old:
            line += f" (I actually called one '{old[0]}')"
        return line

    def misread(self, c: Concept) -> str:
        """Deliberately describe a thing at the WRONG level of the tree - the level it used to live
        at. Only works because both levels and the link between them are kept. 'I want this kind of
        dog' (showing a bear). An approximation of a joke."""
        if not c.parent:
            return c.label
        return f"this kind of {c.parent.label}"

    def concept_for_word(self, word: str) -> Optional[Concept]:
        """Context-resolved; falls back to the best-bound candidate when context can't decide."""
        choice, cands = self.resolve(word)
        if choice is not None or not cands:
            return choice
        return max(cands, key=lambda c: c.word_counts[word])

    def doubles(self) -> dict:
        """Handles that open more than one concept. Noticed, not hidden."""
        words = Counter()
        for c in self.nodes:
            for w, k in c.word_counts.items():
                if k >= MIN_RECUR: words[w] += 1
        return {w: self.concepts_for_word(w) for w, k in words.items() if k > 1}

    # -- introspection ---------------------------------------------------------------------
    def tree(self) -> str:
        lines = []
        def walk(c, depth):
            rel = f" = {c.relation()}" if c.defined_by else ""
            pend = f"  (remembered, unexplained: {c.pending()})" if c.instances else ""
            lines.append("  " * depth + f"- {c.label}  n={c.n}{rel}{pend}")
            for k in c.children: walk(k, depth + 1)
        for r in [c for c in self.nodes if c.parent is None]: walk(r, 0)
        return "\n".join(lines)
