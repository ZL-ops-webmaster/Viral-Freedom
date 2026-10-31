"""The language / structural cortex (design.md §21-26). Handles, structs, frames, episodes, roles, intuition, the web of memberships, doubles and context resolution. It operates over its own state (structs, role_fills, episodes, same) and reads the perceptual tree for grounding (nearest, kinds)."""
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


class Language:
    """A full component now, not a mixin: it OWNS its state and reaches the rest of the
    brain through `self.brain`. English, other natural languages, and coding languages all
    land here -- coding is the understanding of a language and its systems, so it is a language."""
    def __init__(self, brain):
        self.brain = brain                      # coordinator: perception/thinking reached through it
        self.structs: dict = {}                 # structural concepts: handle -> structure
        self.role_fills = Counter()             # (role, filler) -> times seen
        self.same: dict = {}                    # simple equivalences (transform = change)
        self.episodes: dict = {}                # name -> ordered experiences
        self.episode_labels: dict = {}          # name -> narration's labels per position

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
                self.brain.log(f"  LINKED: '{word}' <-> {m.label}  (named after it; connected despite the differences)")

    @staticmethod
    def _is_ancestor(a: Concept, c: Concept) -> bool:
        while c is not None:
            if c is a: return True
            c = c.parent
        return False

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
            "mentioned_in": [n.label for n in self.brain.nodes
                             if n is not c and word in n.label.replace("-", " ").split()],
        }

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
                self.brain.hypotheses.insert(0, h)
                self.brain.log(f"  HEARD '{compound}': I know {' and '.join(parts)} -> I expect a kind of "
                         f"{head.label} that has to do with {', '.join(m.label for m in mods)}")
                return h
        D = self.brain.descriptor_concepts()
        for name, d in sorted(D.items(), key=lambda kv: -len(kv[0])):
            if compound.startswith(name) and len(compound) > len(name):
                rest = compound[len(name):].lstrip("-_ ")
                base = self.concept_for_word(rest)
                if base is None:
                    continue
                p, near, dist = self.brain.imagine(base, [name])
                h = {"question": f"'{compound}' = {name} + {rest}", "base": base, "descriptor": name,
                     "dim": d["dim"], "sign": d["sign"], "proto": p, "novelty": dist, "pull": dist + 10.0,
                     "word": compound}
                self.brain.hypotheses.insert(0, h)                   # it now EXPECTS this thing
                self.brain.log(f"  HEARD '{compound}': I know '{name}' and '{rest}' -> I expect a {name} {rest}")
                return h
        return None

    def concepts_for_word(self, word: str) -> list[Concept]:
        """Every node this handle opens. More than one = a DOUBLE (to / too / two)."""
        return [c for c in self.brain.nodes if c.word_counts.get(word, 0) >= MIN_RECUR]

    def resolve(self, word: str) -> tuple[Optional[Concept], list[Concept]]:
        """Pick which sense of a handle is meant, from CONTEXT: the candidate that most often lights
        up together with what is active right now. Returns (choice or None if truly ambiguous,
        all candidates)."""
        cands = self.concepts_for_word(word)
        if len(cands) <= 1:
            return (cands[0] if cands else None), cands
        active = self.brain.active_nodes()
        if not active:
            return None, cands
        # company, re-read through the CURRENT tree: for each candidate, how often did one of ITS
        # observations arrive right after one of the kinds that are active now? (smoothed one step
        # up the tree). What's already in mind wins outright - it is the most expected thing of all.
        who = [self.brain.nearest(x) for x in self.brain.obs]
        def near_active(c):
            return c in active or c.parent in active or any(a.parent is c for a in active)
        scored = []
        for c in cands:
            s = 1000.0 if c in active else 0.0
            for i, w in enumerate(who):
                if w is c:
                    for j in range(max(0, i - self.brain.WINDOW), i):
                        if who[j] is c: continue
                        if who[j] in active: s += 1.0
                        elif near_active(who[j]): s += 0.5
            scored.append((s, c))
        scored.sort(key=lambda t: -t[0])
        if scored[0][0] == 0 or (len(scored) > 1 and scored[0][0] == scored[1][0]):
            return None, cands                               # nothing in mind tips it: ambiguous
        return scored[0][1], cands

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
        self.brain.log(f"  DEFINED '{word}' = {' + '.join(parts)}")
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
                self.brain.log(f"  '{word}' specialises '{other.name}'  (adds: {', '.join(sorted(set(roles) - set(other.roles)))})")
        self.brain.log(f"  FRAME '{word}' = " + ", ".join(f"{r}={f or '?'}" for r, f in roles.items()))
        return s

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
                    self.brain.log(f"  '{name}' is a progression of '{other}': the same steps, then more")
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
        self.brain.log(f"  '{word}' is a {frame}:  " + ", ".join(f"{r}={f}" for r, f in fillers.items()))
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
        for other in list(self.structs) + [c.label for c in self.brain.nodes]:
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
        self.brain.log(f"  BORROWED: {target} takes '{part}' from {source}"
                 + (f"  ({part} = {' + '.join(self.structs[part].parts)})" if part in self.structs and self.structs[part].parts else ""))
        return t

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
        for c in self.brain.nodes:
            for w, k in c.word_counts.items():
                if k >= MIN_RECUR: words[w] += 1
        return {w: self.concepts_for_word(w) for w, k in words.items() if k > 1}
