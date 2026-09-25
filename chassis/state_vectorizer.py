"""Converts a TrajectorySnapshot (see data/trajectory_schema.py) into a
deterministic float32 feature vector for the ActorCritic policy.

STUB (2026-09-24): design sketched, not yet implemented. Adapted from
worker-prod's StateVectorizer, which builds structural features from a live
SQLite-backed WorkingMemory (aspect completion, execution stats, artifact
type distribution). This supervisor has no such database - it only ever
sees what's actually observable through OpenClaw's hook payloads and the
Durable Object's own tracked session state, captured in a TrajectorySnapshot
before this function ever runs.

Planned feature categories (see the design discussion this chassis is
based on, 2026-09-24):
  - step budget: step_number / max_steps, remaining / max_steps
    (worker-prod's s[40:42], trivial to compute from the snapshot)
  - repetition / stagnation detection: has the same tool + params been
    called N times in a row - worker-prod's stagnation signal, arguably
    the single most directly-reusable idea, since loop detection is close
    to the core of what a supervisor needs
  - tool error telemetry: recent tool-call success/failure/timeout rate,
    built from after_tool_call outcomes (worker-prod's s[42:48] execution-
    stats block, same shape, different data source)
  - risk flags: hard-coded bits for known-risky tool categories (shell
    exec, mass delete, network write) - mirrors worker-prod's
    code_goal/tool_goal/delivery_pending "does this situation need special
    handling" bits
  - goal-drift signal: cosine distance between an embedding of recent
    tool-call context and the stated objective - NOT in worker-prod (it
    has no need for it), reuses Agent Drift Lab's already-proven Cloudflare
    embedding call
  - action history: one-hot recent-action block, same shape as
    worker-prod's HISTORY_LEN x ACTION_COUNT block, resized for this
    vocabulary's ACTION_COUNT (8, see action_space.py)

STATE_DIM is intentionally not pinned down yet - it falls out of how many
of the above categories actually earn their place once real (or synthetic)
trajectories are in hand to test against.
"""
from __future__ import annotations

import numpy as np


def vectorize(snapshot) -> np.ndarray:
    raise NotImplementedError(
        "state_vectorizer.vectorize: not implemented - see this module's docstring for the planned design"
    )
