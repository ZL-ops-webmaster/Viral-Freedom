"""B.R.A.I.N. talks. "If it has no hands, give it a robotic one; if it has no lips, let it text."

A text interface over the structural layer: it learns what you tell it, answers what you ask, asks
back when it hits a word it doesn't know, and volunteers what it's wondering. It keeps a vocabulary
per person, so it tells each person the shortest thing THEY can decompress.

Things it understands (case-insensitive):
  X is a Y                            a kind-of link
  X can Y / X can't Y / X means Y / X is the opposite of Y
  it is hard to Y with X              a negative cue: counts for things that share it, against things that Y well
  the dog ate the cookie              a plain sentence: fills a known verb's roles
  what did X do?                      what X has been the doer of
  X = a + b + c                       a structure (define)
  X: role=?, role=?  (story)          a frame with roles; '?' = a slot
  saw X (who): a > b > c [l1, l2, l3] an episode, in order, with the narration's labels
  what is X?  /  what's X?            breakdown / web / or it asks you
  like X?                             what X is structurally like
  X borrows P from Y                  take one part across
  tell NAME about X                   the shortest description NAME can decompress
  recognize: role=filler, ...         which frame these fillers fit
  how do you feel?  what do you want? think   sleep   remember X?
Run:  python talk.py            (interactive)      python talk.py --script   (the demo conversation)
"""
import json
import os
import re
import sys
from brain import Brain, Struct

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = os.path.join(HERE, "seed_4th_grade.txt")
MEMORY = os.path.join(HERE, "brain_memory.json")
PREPS = (" to ", " with ", " in ", " at ", " on ", " from ", " into ")


class Talker:
    STOP = set("a an the my your his her their this that these those some any of to in on at for with "
               "and or but is are was were be been being have has had do does did made make makes make "
               "what which who how when where why it its i you we they he she me us them very really just "
               "like than then there here also too not no yes hi hello hey".split())

    def __init__(self, brain: Brain):
        self.b = brain
        self.vocab: dict[str, set] = {}          # person -> words they've used (what they can decompress)
        self.unknown: list[str] = []             # words it heard and doesn't know - it will ask, once
        self.asked: set = set()
        self.alias: dict[str, str] = {}          # gave -> give, kid -> child
        self.guessed: dict[str, str] = {}        # word -> the kind it intuited (a guess, until told)
        self.events = 0
        self.said_log: list[str] = []
        brain.log = lambda *a: self.said_log.append(" ".join(map(str, a)).strip())

    def _learned(self, w: str):
        """It just got told what w is: stop wondering about it."""
        if w in self.unknown: self.unknown.remove(w)
        self.asked.discard(w)

    # -- the basic words: learned by being told, from a transcript in its own grammar ----------
    def seed(self, path: str = SEED) -> int:
        n = 0
        if not os.path.exists(path): return 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                self._dispatch("seed", line, line.lower()); n += 1
        self.said_log.clear(); self.unknown.clear()
        self.vocab["kid"] = set(self.b.structs)              # what a kid can decompress: the seed
        return n

    # -- memory on disk: tomorrow continues today ------------------------------------------
    def save(self, path: str = MEMORY):
        b = self.b
        data = {"structs": [{"name": s.name, "parts": s.parts, "links": s.links, "borrowed": s.borrowed,
                             "roles": s.roles, "story": s.story, "is_frame": s.is_frame} for s in b.structs.values()],
                "episodes": b.episodes, "episode_labels": b.episode_labels,
                "role_fills": [[r, f, k] for (r, f), k in b.role_fills.items()],
                "same": {k: sorted(v) for k, v in b.same.items()},
                "vocab": {k: sorted(v) for k, v in self.vocab.items()},
                "unknown": self.unknown, "alias": self.alias, "events": self.events, "guessed": self.guessed}
        with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=1)

    def load(self, path: str = MEMORY) -> bool:
        if not os.path.exists(path): return False
        with open(path, encoding="utf-8") as f: d = json.load(f)
        b = self.b
        for s in d["structs"]:
            b.structs[s["name"]] = Struct(s["name"], s["parts"], [tuple(l) for l in s["links"]],
                                          [tuple(x) for x in s["borrowed"]], s["roles"], s["story"], s["is_frame"])
        b.episodes = d["episodes"]; b.episode_labels = d["episode_labels"]
        for r, f, k in d["role_fills"]: b.role_fills[(r, f)] = k
        b.same = {k: set(v) for k, v in d["same"].items()}
        self.vocab = {k: set(v) for k, v in d["vocab"].items()}
        self.unknown = d["unknown"]; self.alias = d["alias"]; self.events = d.get("events", 0)
        self.guessed = d.get("guessed", {})
        return True

    # -- a plain sentence is a frame with its roles filled ----------------------------------
    def _verb(self, w: str):
        """'gave' -> give, 'eats' -> eat: the frame this word names, if any."""
        w = self.alias.get(w, w)
        for cand in (w, w[:-1] if w.endswith("s") else None, w[:-2] if w.endswith("es") else None,
                     w[:-2] if w.endswith("ed") else None, w[:-3] if w.endswith("ing") else None,
                     w[:-1] + "e" if w.endswith("ed") else None):
            if cand and cand in self.b.structs and self.b.structs[cand].is_frame:
                return cand
        return None

    def _sentence(self, who: str, low: str):
        DET = r"(?:a|an|the|my|your|his|her|their|some)\s+"
        toks = re.sub(r"^" + DET, "", low.replace(",", "")).split()
        if len(toks) < 2: return None
        subj = self.alias.get(toks[0], toks[0])
        if subj == "i": subj = who                           # "I made" - the speaker did
        verb = self._verb(toks[1])
        if verb is None: return None
        rest = " " + " ".join(toks[2:]) + " "
        rest = re.split(r" (?:that|which|who|when|because|so|and) ", rest, 1)[0] + " "   # drop the clause
        liked = re.search(r" like (?:a |an |the )?(\S+) *$", rest)                    # "swings it like a whip"
        if liked: rest = rest[:liked.start()] + " "
        cur, found = rest, []
        for p in PREPS:
            if p in cur:
                before, after = cur.split(p, 1); found.append((before.strip(), after.strip())); cur = " " + after
        obj = (found[0][0] if found else rest).strip()
        extras = [a for _, a in found]
        strip = lambda s: re.sub(r"^" + DET, "", s).strip()
        # "gave rukia a sword": two things after the verb = receiver then thing; the thing comes first in roles
        m = re.match(r"^(\S+)\s+" + DET + r"(.+)$", obj)
        objs = [strip(m.group(2)), m.group(1)] if m else [strip(obj)] * bool(obj)
        fillers = [subj] + objs + [strip(e) for e in extras]
        roles = list(self.b.structs[verb].roles)
        fill = {r: self.alias.get(f, f) for r, f in zip(roles, fillers) if f}
        if liked and len(fill) > 1:                          # the likeness is about the thing, not the doer
            thing = list(fill.values())[1]; self.b._ensure(thing); self.b._ensure(liked.group(1))
            self._link(thing, "like", liked.group(1))
        self.events += 1
        self.b.instantiate(f"{subj}'s {verb} #{self.events}", verb, fill)
        self.said_log.clear()
        for f in fill.values(): self._learned(f)                       # used in a sentence = placed, not unknown
        return (f"ok - {verb}: " + ", ".join(f"{r}={f}" for r, f in fill.items()) +
                "".join(self._intuit(f) for f in fill.values()))

    # -- intuition: it was never told, but it has heard enough ------------------------------
    PROPERTY_KINDS = {"color", "feeling", "number", "size", "shape", "taste", "sound"}
    GUESS_AT = 1.2        # rarity-weighted: about two cues that few things share
    MARGIN = 0.1          # a tie is not a guess

    def _intuit(self, x: str) -> str:
        """A word with no kind, used enough times: guess its kind from how it's used, and say so."""
        b = self.b
        s = b.structs.get(x)
        if s is None: return ""
        told = any(r == "is-a" for r, _ in s.links) and x not in self.guessed
        if told: return ""
        g = b.guess(x, self.GUESS_AT, self.MARGIN)
        if g is None: return ""
        kind, sc, why, tied = g
        old = self.guessed.get(x)
        if old == kind: return ""
        if old:                                              # a guess is held loosely: new evidence can move it
            old_sc = dict((k, v) for k, v, _ in b.intuit(x, top=20)).get(old, 0)
            if sc < old_sc + self.MARGIN and old not in b._kinds_of(kind): return ""
            s.links = [l for l in s.links if l != ("is-a", old)]
            b.structs[old].links = [l for l in b.structs[old].links if l != ("has-kind", x)]
        self._link(x, "is-a", kind); self._link(kind, "has-kind", x); self.guessed[x] = kind
        some = f"some kind of {kind} ({' or '.join(tied)})" if len(tied) > 1 else f"a kind of {kind}"
        return f"\n   ({'actually, ' if old else ''}I think a {x} is {some} - " + "; ".join(why[:3]) + ")"

    def _what_did(self, x: str) -> str:
        x = self.alias.get(x, x)
        done = []
        for s in self.b.structs.values():
            if s.is_frame or not s.roles: continue
            frame = next((g for r, g in s.links if r == "is-a"), None)
            if frame and list(s.roles.values())[0] == x:
                done.append(f"{frame}: " + ", ".join(f"{r}={f}" for r, f in s.roles.items()))
        return ("\n".join("  " + d for d in done)) if done else f"I don't know anything {x} did."

    def _what_is(self, x: str) -> str:
        b = self.b
        s = b.structs.get(x)
        if s is None: return ""
        if s.roles: return b.breakdown(x)
        out = []
        kinds = [g for r, g in s.links if r == "is-a"]
        if kinds:
            chain, k = [], kinds[0]
            while k and len(chain) < 3:
                chain.append(k); k = next((g for r, g in b.structs.get(k, Struct(k, [])).links if r == "is-a"), None)
                if k in chain: k = None
            sure = "I think " if x in self.guessed else ""
            out.append(f"{sure}a {x} is a kind of {chain[0]}" + (f" ({' > '.join(chain[1:])})" if len(chain) > 1 else ""))
            if x in self.guessed:
                g = b.guess(x, 0, self.MARGIN)
                if g: out[-1] += " - nobody told me, but " + "; ".join(g[2][:3])
        elif b.intuit(x):
            res = b.intuit(x)
            close = [k for k, sc, _ in res if res[0][1] - sc < self.MARGIN]
            out.append(f"I'm not sure. maybe a kind of {' or '.join(close)}? " + "; ".join(res[0][2][:2]))
        if s.parts: out.append("it has: " + ", ".join(s.parts))
        only = [g for r, g in s.links if r == "only"]
        can = [g for r, g in s.links if r == "can" and g not in only]
        if only: out.append("it is only ever: " + ", ".join(only))
        if can: out.append("it can: " + ", ".join(can))
        bec = [g for r, g in s.links if r == "becomes"]
        if bec: out.append("otherwise it's " + ", ".join(bec))
        opp = [g for r, g in s.links if r == "opposite"]
        if opp: out.append("the opposite is " + ", ".join(opp))
        subs = [g for r, g in s.links if r == "has-kind"]
        if subs: out.append("kinds: " + ", ".join(subs[:8]))
        return "; ".join(out) if out else ""

    # -- helpers ---------------------------------------------------------------------------
    def _breaks(self, a: str, c: str) -> str:
        """Told 'ice is warm' when ice can only be cold: that's not ice any more. What is it now? Not
        something it has to be told: ice IS water, and water CAN be warm - so warm ice must be water.
        Climb the kinds; the first one that admits the new state is what it became. (Zion: "something
        that is always cold logically must melt or evaporate when it warms up. It should know that.")"""
        b = self.b
        s = b.structs.get(a)
        if s is None: return ""
        for r, y in s.links:
            if r != "only": continue
            opp = [o for rel, o in b.structs.get(y, Struct(y, [])).links if rel == "opposite"]
            if c not in opp: continue
            after = [g for rel, g in s.links if rel == "becomes"]
            if after: return f"then it isn't {a} - {a} is only ever {y}. {c} {a} is {after[0]}."
            for k in b._kinds_of(a):                         # ice > water > drink: who can be warm?
                ks = b.structs.get(k)
                if ks and ("can", c) in ks.links and ("only", y) not in ks.links:
                    self._link(a, "becomes", k); self._link(k, "was", a)
                    self._ensure_change()
                    inst = b.instantiate(f"{a} becoming {k}", "change", {"before": a, "by": c, "after": k})
                    self.said_log.clear()
                    named = self._name_change(inst)
                    return (f"then it isn't {a} - {a} is only ever {y}. it must be {k} now: "
                            f"{a} is {k}, and {k} can be {c}." + (f" that's {named}." if named else ""))
            self.unknown.append(f"{c} {a}"); self.asked.add(f"{c} {a}")
            return f"then it isn't {a} - {a} is only ever {y}. what is {c} {a}?"
        return ""

    def _ensure_change(self):
        b = self.b
        if not (b.structs.get("change") and b.structs["change"].is_frame):
            b.frame("change", {"before": None, "by": None, "after": None}, "a thing is one way, then it is another way")
            self.said_log.clear()

    def _fits(self, x: str, want) -> bool:
        """Does ice fit 'solid'? If it IS it, CAN be it, or is a KIND of it."""
        if want is None or x == want: return True
        s = self.b.structs.get(x)
        return bool(s) and (("can", want) in s.links or ("is-a", want) in s.links or want in self.b._kinds_of(x))

    def _name_change(self, inst) -> str:
        """A change it worked out (ice > water) is MELTING if melt's fixed roles fit: ice is solid,
        water is liquid. Recognition by the fillers' properties - the frame names the event."""
        b = self.b
        for f in b.structs.values():
            if not f.is_frame or f.name == "change" or not {"before", "after"} <= set(f.roles): continue
            if not (f.roles["before"] or f.roles["after"]): continue
            if all(self._fits(inst.roles.get(r), f.roles[r]) for r in ("before", "after")):
                inst.links = [l for l in inst.links if l != ("is-a", "change")]
                self._link(inst.name, "is-a", f.name); self._link(f.name, "has-kind", inst.name)
                return f.name + "ing" if not f.name.endswith("e") else f.name[:-1] + "ing"
        return ""

    def _rename_changes(self, frame: str) -> str:
        """A new name arrives after the fact: do any changes it already knows fit it?"""
        hits = []
        for s in list(self.b.structs.values()):
            if s.is_frame or {"before", "after"} - set(s.roles): continue
            if any(l == ("is-a", "change") for l in s.links) and self._name_change(s): hits.append(s.name)
        return (" so " + "; ".join(hits) + " - that was " + frame + "ing.") if hits else ""

    def _negative(self, a: str, act: str, prefix: str):
        """'can't smash' is a concept of its own: no-smash, the opposite of smash (hard-smash: partly)."""
        neg = prefix + act; b = self.b
        b._ensure(act); b._ensure(neg)
        rel = "opposite" if prefix == "no-" else "partly-opposite"
        self._link(neg, rel, act); self._link(act, rel, neg)
        self._link(a, "can", neg)

    def _link(self, a: str, rel: str, c: str):
        """Being told twice is still one link."""
        if (rel, c) not in self.b.structs[a].links: self.b.structs[a].links.append((rel, c))

    def knows(self, w: str) -> bool:
        return w in self.b.structs or self.b.concept_for_word(w) is not None

    def _note_vocab(self, who: str, text: str):
        self.vocab.setdefault(who, set()).update(re.findall(r"[a-z][a-z0-9\-']*", text.lower()))

    def _drain_log(self) -> str:
        out = "\n".join("   " + l for l in self.said_log); self.said_log.clear(); return out

    # -- the one method: a line in, a reply out ---------------------------------------------
    def handle(self, who: str, text: str) -> str:
        self._note_vocab(who, text)
        t = re.sub(r"^(?:hi|hello|hey|ok|okay|so|well)[!,.\s]+", "", text.strip(), flags=re.I).rstrip(".!").strip()
        low = t.lower()
        low = re.sub(r"^(?:a|an|the|my|your|his|her|their|our|\w+'s) ", "", low)   # "my cat is a pet" is about cat
        reply = self._dispatch(who, t, low)
        log = self._drain_log()
        if log: reply = (reply + "\n" + log).strip()
        ask = self._ask_back()
        return reply + ("\n" + ask if ask else "")

    def _dispatch(self, who, t, low):
        b = self.b
        m = re.match(r"^(.+?) borrows (.+?) from (.+)$", low)
        if m:
            tgt, part, src = m.groups()
            try: b.borrow(tgt, src, part); return f"ok - {tgt} now has {part}, like {src}."
            except ValueError as e: return str(e)
        m = re.match(r"^(?:i )?saw (.+?)(?: \((.+?)\))?: (.+?)(?: \[(.+)\])?$", low)
        if m:
            name, w, steps, labels = m.groups()
            steps = [s.strip() for s in steps.split(">")]
            labels = [l.strip() for l in labels.split(",")] if labels else None
            fr = b.episode(name, steps, labels, who=w or who)
            for s in [name] + steps: self._learned(s)
            return (f"got it - that's {len(b.episodes[name])} time(s) I've seen {name}." +
                    (f" I think I see the pattern: {b.structs[name].story}" if fr else ""))
        m = re.match(r"^recognize: (.+)$", low)
        if m:
            fill = dict(p.split("=") for p in (x.strip() for x in m.group(1).split(",")))
            fill = {k.strip(): v.strip() for k, v in fill.items()}
            res = b.recognize(fill)
            if not res: return "nothing I know has those roles."
            cov, lik, fr, miss, fits = res[0]
            return (f"by shape that's a {fr} ({cov:.0%} of its roles)" + (f", missing {', '.join(miss)}" if miss else "") +
                    f". by experience: {lik:.2f}" + (" - I've never seen anything fill those slots." if lik < 0.15 else ""))
        m = re.match(r"^tell (\w+) about (.+)$", low)
        if m:
            person, x = m.groups()
            return f"to {person}: it's {b.tell(x, self.vocab.get(person, set()))}."
        m = re.match(r"^like (.+?)\??$", low)
        if m:
            x = m.group(1)
            res = b.like(x)
            if not res: return f"{x} isn't like anything I know yet."
            o, other, shared, could = res[0]
            return f"{x} is like {other} (they share {', '.join(shared)}). it could borrow: {', '.join(could) or 'nothing'}."
        m = re.search(r"\bwhat did (\w+) do\??$", low)
        if m:
            return self._what_did(m.group(1))
        m = re.search(r"\bwhat(?:'s| is| are) (?:an? |the )?(.+?)\??$", low)
        if m:
            x = self.alias.get(m.group(1).strip(), m.group(1).strip())
            self.vocab.get(who, set()).difference_update(x.split())   # asking what it is = not holding the handle
            ans = self._what_is(x)
            if ans: return ans
            w = b.web(x)
            if w: return f"{x}: is {' > '.join(w['is']) or '?'}; kinds: {', '.join(w['kinds']) or '-'}; linked to: {', '.join(w['linked']) or '-'}"
            if x in b.structs: return f"I've heard of {x}, but I only have the word. what is it?"
            if x not in self.unknown: self.unknown.append(x)
            self.asked.add(x)                                # this reply IS the question; don't ask twice
            return f"I don't know what {x} is. what is it?"
        if low in ("how do you feel", "how do you feel?"):
            _, mood = b.feel()
            if self.unknown: mood = f"curious: {len(self.unknown)} word(s) I don't know yet ({self.unknown[0]} first)"
            return f"{mood}."
        if low in ("what do you want", "what do you want?"):
            d = b.desires(); return ("I want: " + ", ".join(w.name for w in d)) if d else "nothing yet. nothing has felt like anything."
        if low == "think":
            r = b.think(); qs = [h["question"] for h in r["questions"][:3]] + [f"what's a {w}?" for w in self.unknown[:2]]
            self.asked.difference_update(self.unknown)       # thinking brings a dropped question back
            return f"({r['mood_before']})" + (" I'm wondering: " + "; ".join(qs) if qs else " nothing to wonder about.")
        if low == "sleep":
            r = b.sleep(); return f"slept. let go of {r['forgot']} memories; kept {r['kept']}."
        m = re.match(r"^(?:do you )?remember (.+?)\??$", low)
        if m:
            x = m.group(1)
            if x in b.structs: return f"yes - {x}" + (f": {b.breakdown(x)}" if b.structs[x].roles else "")
            return f"no, I don't remember a {x}."
        m = re.match(r"^(.+?): (.+?)(?: \((.+)\))?$", t)
        if m and "=" in m.group(2):
            name, body, story = m.group(1).lower(), m.group(2), m.group(3) or ""
            roles = {}
            for p in body.split(","):
                r, v = [x.strip() for x in p.split("=")]
                roles[r.lower()] = None if v == "?" else v.lower()
            b.frame(name, roles, story); self._learned(name)
            return f"ok - {name} is a frame with {len(roles)} roles."
        m = re.match(r"^(.+?) = (.+)$", low)
        if m and "+" in m.group(2):
            name, parts = m.group(1), [p.strip() for p in m.group(2).split("+")]
            b.define(name, parts)
            for s in [name] + parts: self._learned(s)
            return f"ok - {name} = {' + '.join(parts)}."
        m = re.match(r"^(.+?) is the opposite of (.+)$", low)
        if m:
            a, c = m.groups(); b._ensure(a); b._ensure(c)
            self._link(a, "opposite", c); self._link(c, "opposite", a)
            self._learned(a); self._learned(c)
            return f"ok - {a} and {c} are opposites."
        m = re.match(r"^(.+?) means (.+)$", low)
        if m:
            a, c = m.groups(); b._ensure(c)
            self.alias[a] = c; b.equate(a, c); self._learned(a)
            return f"ok - {a} is another word for {c}."
        m = re.match(r"^(.+?) (?:can't|cannot|can not) (\w+)(?: .*)?$", low)     # "can't smash a rock": the act is smash
        if m:
            a, c = m.groups(); b._ensure(a)
            self._negative(a, c, "no-"); self._learned(a)
            return f"ok - a {a} can't {c}." + self._intuit(a)
        m = re.match(r"^(?:it is|it's) (?:hard|difficult) to (\w+) (?:with|using) (?:an? |the )?(.+)$", low)
        m2 = re.match(r"^(.+?) is (?:hard|difficult) to (\w+) with$", low)
        if m or m2:
            c, a = m.groups() if m else reversed(m2.groups())
            b._ensure(a)
            self._negative(a, c, "hard-"); self._learned(a)
            return f"ok - it's hard to {c} with a {a}." + self._intuit(a)
        m = re.match(r"^(.+?) (?:can only be|is only|is always|can only) (\w+)$", low)
        if m:
            a, c = m.groups(); b._ensure(a); b._ensure(c)
            self._link(a, "can", c); self._link(a, "only", c); self._learned(a)
            return f"ok - {a} is only ever {c}."
        m = re.match(r"^(\w+) (?:is|means) (?:going|changing|turning) from (?:a |an )?(\w+) (?:to|into) (?:a |an )?(\w+)$", low)
        if m:
            name, before, after = m.groups()
            stem = name[:-3] if name.endswith("ing") else name                 # melting -> melt
            if stem != name: self.alias[name] = stem; b.equate(name, stem)
            self._ensure_change()
            for n in (before, after): b._ensure(n)
            b.frame(stem, {"before": before, "by": None, "after": after}, f"something {before} becomes {after}")
            general = "transform" if "transform" in b.structs else "change"      # Zion: "melting is transform"
            self._link(stem, "is-a", general); self._link(general, "has-kind", stem)
            self.said_log.clear(); self._learned(stem); self._learned(name)
            return f"ok - {stem}: {before} > {after}." + self._rename_changes(stem)
        m = re.match(r"^if (.+?) is (\w+),? (?:it is|it's|then it is|it becomes) (?:a |an )?(.+)$", low)
        if m:
            a, by, after = m.groups()
            for n in (a, by, after): b._ensure(n)
            self._link(a, "becomes", after); self._link(after, "was", a)
            if not (b.structs.get("change") and b.structs["change"].is_frame):
                b.frame("change", {"before": None, "by": None, "after": None}, "a thing is one way, then it is another way")
            b.instantiate(f"{a} becoming {after}", "change", {"before": a, "by": by, "after": after})
            self.said_log.clear()
            return f"ok - {by} {a} isn't {a}, it's {after}."
        m = re.match(r"^(.+?) can (?:be )?(\w+)(?: .*)?$", low) or re.match(r"^(.+?) is (\w+)$", low)   # "soup can be hot" / "soup is hot"
        if m:
            a, c = m.groups()
            # "ice is water": no article, but water is a THING it knows (it has kinds, parts or a kind) - a kind, not a property
            thing = (c in b.structs and any(r in ("is-a", "has-kind", "has-part") for r, _ in b.structs[c].links)
                     and not any(r == "opposite" for r, _ in b.structs[c].links)          # cold has an opposite: a property
                     and not set(b._kinds_of(c)) & self.PROPERTY_KINDS)                   # white is a color: a property
            if " is " in low and thing:
                b._ensure(a); self._link(a, "is-a", c); self._link(c, "has-kind", a); self._learned(a)
                return f"ok - {a} is a kind of {c}." + self._intuit(a)
            broke = self._breaks(a, c)
            if broke: return broke
            b._ensure(a); b._ensure(c)
            self._link(a, "can", c); self._learned(a)
            return (f"ok - {a} is {c}." if " is " in low else f"ok - a {a} can {c}.") + self._intuit(a)
        m = re.match(r"^(.+?) is an? (.+)$", low)
        if m:
            a, c = m.groups()
            b._ensure(a); b._ensure(c)
            was = self.guessed.pop(a, None)
            if was and was != c:                                 # told: the guess gives way
                b.structs[a].links = [l for l in b.structs[a].links if l != ("is-a", was)]
                b.structs[was].links = [l for l in b.structs[was].links if l != ("has-kind", a)]
            self._link(a, "is-a", c); self._link(c, "has-kind", a)
            self._learned(a)
            verdict = "" if not was else (" - right." if was == c else " - close." if c in b._kinds_of(was) else " - wrong, noted.")
            return f"ok - {a} is a kind of {c}." + (f" (I guessed {was}{verdict})" if was else "")
        m = re.match(r"^(.+?) has an? (.+?)(?: on (?:the |its )?(.+))?$", low)
        if m:
            a, p, where = m.groups()
            for n in (a, p, where):
                if n: b._ensure(n)
            self._link(a, "has-part", p)
            if where: self._link(a, "has-part", where); self._link(where, "has-part", p)
            self._learned(a)
            return f"ok - a {a} has a {p}" + (f" (on its {where})" if where else "") + "." + self._intuit(a)
        said = self._sentence(who, low)                      # a plain sentence: a frame, roles filled
        if said: return said
        # free text: the things in it are what follow a determiner ("a gauntlet", "the sword");
        # verbs and function words are not concepts. Note what it doesn't know; ask later, once.
        nouns = re.findall(r"\b(?:a|an|the|my|your|his|her|their)\s+([a-z][a-z\-']+)", low)
        new = [w for w in nouns if w not in self.STOP and not self.knows(w) and w not in self.unknown]
        for w in new[:2]: self.unknown.append(w)
        return "ok." if not new else f"ok. ({', '.join(new[:2])} - new to me)"

    def _ask_back(self) -> str:
        """Ask about one thing it doesn't know - once. If nobody answers, it drops it (and think()
        brings it back: a dropped question regenerates when it idles, §14)."""
        for w in self.unknown:
            if w not in self.asked:
                self.asked.add(w)
                return f"-> what's a {w}?"
        return ""


SCRIPT = [
    ("zion", "what is a dog?"),
    ("zion", "what's a whale?"),
    ("zion", "the dog ate the cookie"),
    ("zion", "ichigo gave rukia a sword"),
    ("zion", "what did ichigo do?"),
    ("zion", "what is give?"),
    ("zion", "the cook cut the bread with a knife"),
    ("zion", "the knight cut the rope with his sword"),
    ("zion", "ichigo cut the hollow with his zanpakuto"),
    ("zion", "what is a zanpakuto?"),
    ("zion", "rukia's zanpakuto has a tassel on the handle"),
    ("zion", "renji swings his zanpakuto like a whip"),
    ("zion", "what is a zanpakuto?"),
    ("zion", "the woodsman chopped the log with his tomahawk"),
    ("zion", "the woodsman cut the rope with his tomahawk"),
    ("zion", "a tomahawk has a handle"),
    ("zion", "what is a tomahawk?"),
    ("zion", "it is hard to slice with a tomahawk"),
    ("zion", "a kukri can cut"),
    ("zion", "a kukri can't smash a rock"),
    ("zion", "what is a kukri?"),
    ("zion", "a hatchet can smash a coconut"),
    ("zion", "what is a hatchet?"),
    ("zion", "a hatchet can chop"),
    ("zion", "what is a hatchet?"),
    ("zion", "what is ice?"),
    ("zion", "the ice is warm"),
    ("zion", "melting is going from solid to liquid"),
    ("zion", "what did ice do?"),
    ("zion", "what is snow?"),
    ("zion", "snow is cold"),
    ("zion", "snow is white"),
    ("zion", "what is snow?"),
    ("zion", "senbonzakura = sword + petals + projectile + safe-zone + reach + many + controlled"),
    ("zion", "safe-zone = region + no-entry + owner-override"),
    ("zion", "what is senbonzakura?"),
    ("zion", "I made a gauntlet that shoots a fan of darts when you gesture"),
    ("zion", "gauntlet = hand + gesture + darts + projectile + fan + many + controlled"),
    ("zion", "like gauntlet?"),
    ("zion", "gauntlet borrows safe-zone from senbonzakura"),
    ("zion", "bankai: weapon=?, command=?, transform=?, new-form=? (a transformation of a weapon by a command into a new form)"),
    ("zion", "saw shikai (ichigo): zangetsu-sealed > release > blade-widens > zangetsu-cleaver [weapon, command, transform, new-form]"),
    ("zion", "saw shikai (rukia): sode-sealed > dance > blade-frosts > sode-white [weapon, command, transform, new-form]"),
    ("zion", "recognize: weapon=sword, command=button, transform=changes-form, new-form=spiked-gauntlet"),
    ("maya", "hi! what is bankai?"),
    ("zion", "tell maya about bankai"),
    ("zion", "tell zion about bankai"),
    ("zion", "tell kid about bankai"),
    ("zion", "what's a cookie-cutter shark?"),
    ("zion", "cookie-cutter shark is a shark"),
    ("zion", "how do you feel?"),
    ("zion", "think"),
]


def run_script():
    tk = Talker(Brain(log=print))
    n = tk.seed()
    print(f"(seeded with {n} lines of basic words - about a 4th grader's)")
    for who, line in SCRIPT:
        print(f"\n{who}: {line}")
        print("brain:", tk.handle(who, line))


def main():
    tk = Talker(Brain(log=print))
    n = tk.seed()
    remembered = tk.load()
    who = input("who's there? ").strip() or "you"
    print(f"brain: hi{', ' + who if who != 'you' else ''}. " +
          ("I remember our last talk. " if remembered else f"I know about {n} basic things. ") +
          "teach me, ask me. (ctrl-c to stop)")
    while True:
        try:
            line = input(f"{who}: ")
        except (EOFError, KeyboardInterrupt):
            tk.save(); print("\nbrain: bye. I'll remember."); break
        if line.strip():
            print("brain:", tk.handle(who, line))
            tk.save()


if __name__ == "__main__":
    run_script() if "--script" in sys.argv else main()
