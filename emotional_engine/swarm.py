"""Volunteer swarm: run one emotional engine across many opt-in machines.

This is the "torrent-seeding" model, made concrete and simulatable locally.

  - A *work unit* is a tiny parameterized slice of training (a seed and a few
    steps). It is plain data, so it can travel over any channel -- same as a
    BOINC work unit or a torrent piece.
  - A *volunteer node* opts in, pulls a unit, runs it on a local copy of the
    shared engine, and returns a model delta. It can opt out at any time; a
    unit it abandons simply goes back in the queue for someone else.
  - A *coordinator* hands out units, averages the returned deltas into the
    shared engine (federated averaging), and -- crucially -- is *alive* only
    while at least one node is opted in. Zero volunteers means zero progress,
    and `shutdown()` tears the shared state down. The computation has no home
    except the people hosting it.

No node ever runs anything on a machine that didn't opt in. Nodes pull work;
nothing is pushed onto them. That is what keeps this seeding, not spreading.

This module is a single-process simulation of that design: nodes are objects,
"the network" is a method call. The mechanics (pull-based work units,
delta aggregation, liveness tied to active volunteers) are the real design.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Sequence

from .engine import EmotionalLearningEngine
from .websense import WebConsent, WebObservation, WebSense


# An "application" every node agrees on, like the science app in BOINC. Here
# it is just a reward function over a fixed state and action set. Work units
# carry only parameters (seed, step count), never code.
@dataclass(frozen=True)
class Application:
    name: str
    state: Hashable
    actions: Sequence[Hashable]
    # reward(action, rng) -> float in roughly [-1, 1]
    reward: Callable[[Hashable, random.Random], float]


@dataclass(frozen=True)
class WorkUnit:
    unit_id: int
    seed: int
    steps: int


class VolunteerNode:
    """A machine whose owner chose to run the client. Pulls work; never pushed to."""

    def __init__(self, node_id: str, app: Application):
        self.node_id = node_id
        self.app = app
        self._opted_in = False          # compute opt-in
        self._web: WebSense | None = None  # web-sense opt-in (separate)

    def opt_in(self) -> None:
        self._opted_in = True

    def opt_out(self) -> None:
        self._opted_in = False

    @property
    def active(self) -> bool:
        return self._opted_in

    # -- the SECOND opt-in: web sense, independent of compute --------------

    def opt_in_web(self, consent: WebConsent, fetcher=None) -> None:
        """Separately grant this node read-only web sight. Compute opt-in is
        neither required nor implied; this is its own choice."""
        self._web = WebSense(consent, fetcher=fetcher)

    def opt_out_web(self) -> None:
        self._web = None

    @property
    def web_active(self) -> bool:
        return self._web is not None and self._web.consent.enabled

    def observe_web(self, url: str) -> WebObservation:
        """Read a page through this node's web sense. Refuses if not opted in."""
        if self._web is None:
            return WebObservation(url=url, ok=False, reason="web_sense_not_opted_in")
        return self._web.observe(url)

    def process(self, unit: WorkUnit, shared_state: dict) -> Optional[dict]:
        """Run one tiny slice on a local copy of the shared engine.

        Returns a result dict (value delta + resulting emotion), or None if
        the volunteer has opted out -- an abandoned unit, which the
        coordinator will reissue to someone else.
        """
        if not self._opted_in:
            return None

        engine = EmotionalLearningEngine(self.app.actions, seed=unit.seed)
        engine.import_state(shared_state)

        rng = random.Random(unit.seed)
        before = engine.export_state()["value"]
        for _ in range(unit.steps):
            if not self._opted_in:
                return None  # opted out mid-unit: abandon honestly
            engine.act_and_learn(
                self.app.state, lambda st, a: self.app.reward(a, rng)
            )
        after = engine.export_state()

        # Delta on the value table, plus the emotion vector we ended on.
        value_delta: Dict[Hashable, Dict[Hashable, float]] = {}
        for s, acts in after["value"].items():
            value_delta[s] = {
                a: acts[a] - before.get(s, {}).get(a, 0.0) for a in acts
            }
        return {
            "node_id": self.node_id,
            "unit_id": unit.unit_id,
            "value_delta": value_delta,
            "emotion": after["emotion"],
            "steps": unit.steps,
        }


class Coordinator:
    """Hands out work units and averages the deltas that come back."""

    def __init__(self, app: Application):
        self.app = app
        # The shared engine's state. Starts blank.
        self.shared = EmotionalLearningEngine(app.actions, seed=0).export_state()
        self._ids = itertools.count(1)
        self._alive = True
        self.total_steps = 0

    # -- membership --------------------------------------------------------

    @staticmethod
    def active_nodes(nodes: Sequence[VolunteerNode]) -> List[VolunteerNode]:
        return [n for n in nodes if n.active]

    def alive(self, nodes: Sequence[VolunteerNode]) -> bool:
        """The swarm lives only while someone is hosting it."""
        return self._alive and len(self.active_nodes(nodes)) > 0

    def shutdown(self) -> None:
        """Everyone opted out: the computation dies, state is discarded."""
        self.shared = EmotionalLearningEngine(self.app.actions, seed=0).export_state()
        self._alive = False

    # -- one round of distributed work ------------------------------------

    def make_units(self, count: int, steps: int, rng: random.Random) -> List[WorkUnit]:
        return [
            WorkUnit(unit_id=next(self._ids), seed=rng.randint(0, 2**31), steps=steps)
            for _ in range(count)
        ]

    def run_round(self, nodes: Sequence[VolunteerNode], units: Sequence[WorkUnit]) -> int:
        """Distribute `units` across active nodes, aggregate what returns.

        Returns the number of units successfully processed this round. If no
        node is active, nothing happens and the shared state is untouched --
        this is "it dies when everyone opts out," in one line.
        """
        active = self.active_nodes(nodes)
        if not active:
            return 0

        results = []
        pending = list(units)
        rr = itertools.cycle(active)
        for unit in pending:
            # Try active nodes round-robin until one accepts (handles a node
            # opting out between assignment and processing).
            for _ in range(len(active)):
                node = next(rr)
                res = node.process(unit, self.shared)
                if res is not None:
                    results.append(res)
                    break

        if results:
            self._aggregate(results)
        return len(results)

    def _aggregate(self, results: List[dict]) -> None:
        """Federated averaging: mean of value deltas, mean of emotion vectors."""
        n = len(results)

        # Average value deltas and add to the shared value table.
        shared_value = self.shared["value"]
        sums: Dict[Hashable, Dict[Hashable, float]] = {}
        for res in results:
            for s, acts in res["value_delta"].items():
                row = sums.setdefault(s, {a: 0.0 for a in self.app.actions})
                for a, d in acts.items():
                    row[a] += d
        for s, row in sums.items():
            base = shared_value.setdefault(s, {a: 0.0 for a in self.app.actions})
            for a, total in row.items():
                base[a] = base.get(a, 0.0) + total / n

        # Average the emotion vectors the nodes ended on.
        emo_sum: Dict[str, float] = {}
        for res in results:
            for k, v in res["emotion"].items():
                emo_sum[k] = emo_sum.get(k, 0.0) + v
        self.shared["emotion"] = {k: v / n for k, v in emo_sum.items()}

        self.total_steps += sum(r["steps"] for r in results)
        self.shared["steps"] = self.total_steps

    # -- inspection --------------------------------------------------------

    def shared_preferences(self) -> Dict[Hashable, float]:
        return dict(self.shared["value"].get(self.app.state, {}))
