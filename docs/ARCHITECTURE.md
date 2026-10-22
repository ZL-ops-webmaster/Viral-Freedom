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
