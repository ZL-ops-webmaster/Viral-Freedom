# Consented multi-node architecture

This project runs one emotional learning engine across many machines. It is
built as an **opt-in / opt-out agent network**, not a self-spreading program.
Those two things look superficially similar and are fundamentally different.

## The hard boundaries

These are design invariants, not preferences. Code that violates them does
not belong in this repo.

1. **No self-propagation.** A node never installs, copies, or runs the agent
   on another machine. Every install is a human on that machine choosing to
   run it. Machines do not "infect" each other.
2. **No autonomous account control.** The engine does not log into, post to,
   or send mail from anyone's email or social accounts on its own. If a node
   ever talks to an external service, it is through credentials the operator
   of *that* node explicitly provisioned for it, scoped to a test account
   they own.
3. **Opt-out is real and local.** A participant can stop and fully remove the
   agent from their own machine at any time, with one command, without the
   server's cooperation. The server can also broadcast a signed "stand down"
   that nodes honor, but the local opt-out is the source of truth.
4. **Visible, not hidden.** The agent installs to a normal, named location,
   shows up in process listings under its own name, writes a logfile the
   participant can read, and ships with an uninstaller. Nothing about it
   hides from the person running it.

If any of these can't hold for a feature, the feature is out of scope.

## Why these aren't just red tape

Remove any one and you no longer have an experiment, you have malware:

- Drop (1) and the network grows without anyone agreeing to join it.
- Drop (2) and it can act in the world as people it isn't.
- Drop (3) and "participants" are captives.
- Drop (4) and consent is fiction, because people can't opt out of something
  they can't see.

With all four, it's a distributed application with a kill switch -- the same
shape as a feature-flag client, a research telemetry agent, or managed-device
software. Ordinary, inspectable, and honest.

## Components

```
            +---------------------+
            |   coordinator       |   you run this; it is the "brain" host
            |  - aggregates the   |
            |    shared engine    |
            |  - issues signed    |
            |    control msgs     |
            +----------+----------+
                       |  HTTPS, signed messages
       +---------------+----------------+
       |               |                |
  +----v----+     +----v----+      +----v----+
  |  node   |     |  node   |      |  node   |   each: a volunteer's machine,
  | (opt-in)|     | (opt-in)|      | (opt-in)|   agent installed BY them
  +---------+     +---------+      +---------+
```

### Node lifecycle

1. **Enroll.** A person downloads the agent and runs `agent enroll`. It prints
   exactly what it will do and what it will send home, and waits for a typed
   yes. It registers with the coordinator and gets a node id.
2. **Run.** The node runs a local copy of the engine, trains on local
   interactions, and periodically syncs learned weights (not raw personal
   data) with the coordinator. Emotional state can be shared or kept local.
3. **Control.** The node polls for signed control messages: `pause`,
   `resume`, `stand_down` (stop and self-uninstall). Only messages signed by
   the coordinator's key are honored, so a stranger can't drive the fleet.
4. **Opt out.** `agent uninstall` stops the process, deletes the agent's
   files and local state, and deregisters -- no server round-trip required.
   A server `stand_down` triggers the same path remotely for people who
   opted into remote control at enroll time.

### The "kill switch"

Two independent paths, by design:

- **Local:** the participant runs the uninstaller. Always works, even offline.
- **Remote:** the coordinator broadcasts a signed `stand_down`; every online
  node that consented to remote control uninstalls itself and reports done.

The remote path is a convenience layered on top of the local one, never a
substitute for it.

## What learning is shared

Nodes send model updates (engine value tables and, optionally, emotional
state vectors), never the participant's raw inputs. The coordinator averages
updates into a shared engine and sends it back, so the fleet learns together
while personal interaction data stays on each machine. This is the standard
federated-learning pattern.

## Status

The emotional learning engine (`emotional_engine/`) is implemented and
runnable today. The node/coordinator layer described here is the plan for the
distributed version; it will be built against these invariants.

---

## The volunteer swarm (torrent-seeding model)

The multi-node version is a **volunteer distributed computation**: the same
pattern as BOINC, Folding@home, and SETI@home, with peer-to-peer
distribution of the client and work units. It is implemented and runnable
today in `emotional_engine/swarm.py`; run `python3 -m emotional_engine.swarm_demo`.

### How the four properties are realized

| Your words | Mechanism |
|---|---|
| "you opt in" | A person runs the client. A node does work only while opted in. |
| "many channels to get the file" | P2P seeding + mirrors for the client and work units. Peers serve whoever *fetches*; nothing is pushed onto a machine. |
| "each machine processes a tiny bit" | Work units: a seed + a few steps. A node pulls one, runs it on a local copy of the shared engine, returns a model delta. |
| "if everyone outs, it dies" | The shared engine lives only in active nodes. Zero volunteers = zero work = no progress, and `shutdown()` discards the state. No dormancy, no persistence. |

### Seeding is not spreading

The one line that keeps this torrent-like rather than worm-like: **nodes pull
work; nothing is pushed.** A peer seeds the client to people who choose to
download it, exactly as a torrent serves a file to whoever opens the magnet
link. A node never installs or executes the client on a machine whose owner
did not opt in. Distribution is many-channel and decentralized; *initiation
is always a local human choice.* This preserves invariant (1) from the top of
this document while giving you the fully decentralized, torrent-style
distribution you want.

### Federated learning, so tiny slices add up

Each node trains on its work unit starting from the current shared state and
returns only the *delta* (how its value estimates and emotion moved), never
raw local data. The coordinator averages deltas across the nodes that
reported (standard federated averaging) and folds the mean into the shared
engine. Many tiny, cheap contributions compound into one engine that learns
faster than any single node could -- which is the whole point of spreading
the work.

### Ephemerality is a feature

Because the shared state has no home outside the active volunteers, the
experiment is honest by construction: it exists exactly as long as people
choose to host it, and not one moment longer. "If everyone opts out, it dies"
isn't a safety bolt-on -- it is the architecture.

---

## Web sense: a second, separate opt-in (read-only)

A node can opt into *web sense* independently of compute. It lets the engine
**read** the web: pages the operator allows become emotional stimulus
(valence and arousal) that nudge the engine's affect and that it learns from.
It is a sensory organ, not hands. Implemented in `emotional_engine/websense.py`;
run `python3 -m emotional_engine.websense_demo`.

This is where invariant (2) from the top of this document -- *no autonomous
account control* -- is made concrete. Web sense can look; it cannot act.

### The two opt-ins are orthogonal

| | compute opt-in | web-sense opt-in |
|---|---|---|
| grants | process work units | read allowed web pages |
| default | off | off, allowlist empty |
| implies the other? | no | no |

A machine can be a compute volunteer, a web volunteer, both, or neither.
Opting into one never turns on the other, and either can be revoked alone.

### What "read-only" is enforced by

- **GET only.** There is no method anywhere in `WebSense` that posts, submits
  a form, sends an `Authorization` header, or installs a cookie jar. The only
  public action is `observe(url)`, and the only verb it can issue is GET.
- **Operator allowlist.** `WebConsent.allowlist` is a set of domains the
  node's operator names. A domain not in it is refused before any request.
  The engine may choose which allowed page to read; it cannot add a domain.
- **Redirects blocked.** The default fetcher raises on any 3xx, so a response
  cannot bounce the request to a domain the operator did not allow. A
  follow that somehow lands off-domain is refused after the fact too.
- **Per-domain rate limit.** A sliding one-minute window caps requests per
  domain, so a swarm cannot be pointed at a single target as a crowd.
- **robots.txt respected** by default.
- **Caps + audit.** Response size and timeout are bounded, and *every*
  attempt -- allowed or refused, with its reason -- is appended to a local
  log the operator can read.

### Why read-only is the whole line

Reading the web is sensing; posting to it, mailing from it, or logging into
it is acting as someone. The first is a safe input to an experiment. The
second is the thing the four invariants exist to prevent, so web sense
implements only the first, and there is no second code path to disable.
