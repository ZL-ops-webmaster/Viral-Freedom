# Viral-Freedom

An **emotional learning engine** and a design for running it across many
machines as a *consented, opt-in network* -- not a self-spreading program.

## The idea

A learner whose affective state is not decoration but a set of **weighted
values that steer the learning itself**. Its mood changes how boldly it
explores and how sharply it learns, and every outcome feeds back into its
mood. The long-term goal is to run one shared engine across many volunteers'
machines so it learns from many streams at once.

## Two parts

1. **`emotional_engine/`** -- the core, implemented and runnable now.
2. **`docs/ARCHITECTURE.md`** -- the plan for the multi-node version, built
   around opt-in enrollment and a real local opt-out.

## How the engine feels

- Emotions are intensities in `[0, 1]`: joy, trust, fear, surprise, sadness,
  anger, anticipation. They decay toward a baseline over time.
- **Valence** (pleasant/unpleasant) tilts which actions look appealing.
- **Arousal** (calm/activated) scales the learning rate.
- **Fear** narrows exploration; **joy/anticipation** widen it.
- Each outcome both updates value estimates and nudges emotion, so the two
  loops drive each other.

## Run the demo

```bash
python3 -m emotional_engine.demo
```

A forager learns which of three patches to prefer. You'll watch its value
estimates converge while its dominant emotion shifts from surprise to joy as
the good patch pays off.

## Run the volunteer swarm

```bash
python3 -m emotional_engine.swarm_demo
```

Five volunteers opt in and each process tiny work units; the shared engine
learns from their pooled deltas. Then most opt out and the rest carry it; then
everyone opts out and it dies. This is the torrent-seeding / volunteer-compute
model (BOINC- and BitTorrent-style) -- see `docs/ARCHITECTURE.md`.

## Run the tests

```bash
python3 -m unittest discover -s tests -v
```

## On the "virus" framing

This is deliberately **not** a virus. A virus spreads to machines without
their owners' agreement and hides itself. This project does the opposite:
every machine joins because a person on it chose to, the agent is visible and
named, and anyone can remove it from their own machine at any time with one
command. The four invariants that keep it that way are spelled out at the top
of `docs/ARCHITECTURE.md` -- no self-propagation, no autonomous account
control, a real local opt-out, and nothing hidden. They're load-bearing.

## Layout

```
emotional_engine/
  emotions.py     affective state: weighted values, decay, valence, arousal
  engine.py       the learner; emotion modulates exploration and learning rate
  demo.py         runnable foraging demo
  swarm.py        volunteer swarm: work units, opt-in nodes, coordinator
  swarm_demo.py   runnable opt-in / opt-out swarm demo
tests/
  test_engine.py
  test_swarm.py
docs/
  ARCHITECTURE.md   volunteer-swarm design + hard boundaries
```
