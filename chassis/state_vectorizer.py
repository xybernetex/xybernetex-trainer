"""Converts a TrajectorySnapshot (see data/trajectory_schema.py) into a
deterministic float32 feature vector for the ActorCritic policy.

Adapted from worker-prod's StateVectorizer, which builds structural
features from a live SQLite-backed WorkingMemory (aspect completion,
execution stats, artifact type distribution). This supervisor has no such
database - it only ever sees what's actually observable through OpenClaw's
hook payloads and the Durable Object's own tracked session state, captured
in a TrajectorySnapshot before this function ever runs. Zero network calls,
zero I/O side effects - see chassis/__init__.py.

Structural layout (indices into the first _STRUCT_DIM slots):
  [0]  step_number / max_steps
  [1]  (max_steps - step_number) / max_steps           - remaining budget
  [2]  is_repeat_of_previous_call (0/1)                - same tool+params as the prior call
  [3]  consecutive_repeat_count, normalized             - the loop-killer signal
  [4]  recent success_rate (last up to _RECENT_WINDOW tool calls)
  [5]  recent timeout_rate
  [6]  recent error_rate (failed, not timed out)
  [7]  consecutive_error_count, normalized             - trailing errors specifically
  [8]  is_risky_tool (0/1)                             - most recent call matches a risky category
  [9]  cost_so_far / cost_budget, capped at 1.0
  [10] remaining_budget_fraction
  [11] goal_drift_score, or 0.0 if not computed for this snapshot

[_STRUCT_DIM : STATE_DIM] one-hot recent-action history, HISTORY_LEN steps
x ACTION_COUNT actions, oldest first, flattened - same shape as worker-prod's
own action-history block, resized for this vocabulary's ACTION_COUNT.
"""
from __future__ import annotations

import numpy as np

from .action_space import ActionSpace

ACTION_COUNT: int = ActionSpace.N
HISTORY_LEN: int = 16
_STRUCT_DIM: int = 12
STATE_DIM: int = _STRUCT_DIM + HISTORY_LEN * ACTION_COUNT

_RECENT_WINDOW: int = 8
_MAX_CONSECUTIVE_REPEAT: float = 5.0
_MAX_CONSECUTIVE_ERRORS: float = 5.0

# Canonical source for "is this tool call risky" - teacher/heuristic_teacher.py
# imports this rather than redefining its own copy, the same way worker-prod's
# rollout_generator.py imports keyword helpers from its own rl_chassis.py.
_RISKY_TOOL_KEYWORDS: frozenset[str] = frozenset({
    "exec", "shell", "bash", "powershell", "eval", "run_command",
    "delete", "remove", "rm", "drop", "truncate", "purge",
    "payment", "charge", "transfer", "withdraw",
    "send_email", "post", "publish", "deploy",
})


def is_risky_tool(tool_name: str) -> bool:
    name = tool_name.lower()
    return any(keyword in name for keyword in _RISKY_TOOL_KEYWORDS)


def _same_call(a, b) -> bool:
    return a.tool_name == b.tool_name and a.params == b.params


def _consecutive_repeat_count(tool_history: list) -> int:
    if not tool_history:
        return 0
    last = tool_history[-1]
    count = 0
    for call in reversed(tool_history):
        if _same_call(call, last):
            count += 1
        else:
            break
    return count - 1  # the last call itself doesn't count as a "repeat"


def _consecutive_error_count(tool_history: list) -> int:
    count = 0
    for call in reversed(tool_history):
        if not call.success:
            count += 1
        else:
            break
    return count


def _build_structural(snapshot) -> np.ndarray:
    s = np.zeros(_STRUCT_DIM, dtype=np.float32)

    horizon = max(snapshot.max_steps, 1)
    s[0] = snapshot.step_number / horizon
    s[1] = (snapshot.max_steps - snapshot.step_number) / horizon

    history = snapshot.tool_history
    if len(history) >= 2:
        s[2] = 1.0 if _same_call(history[-1], history[-2]) else 0.0
    s[3] = min(_consecutive_repeat_count(history) / _MAX_CONSECUTIVE_REPEAT, 1.0)

    recent = history[-_RECENT_WINDOW:]
    if recent:
        s[4] = sum(1 for c in recent if c.success) / len(recent)
        s[5] = sum(1 for c in recent if c.timed_out) / len(recent)
        s[6] = sum(1 for c in recent if not c.success and not c.timed_out) / len(recent)
    s[7] = min(_consecutive_error_count(history) / _MAX_CONSECUTIVE_ERRORS, 1.0)

    s[8] = 1.0 if (history and is_risky_tool(history[-1].tool_name)) else 0.0

    if snapshot.cost_budget > 0:
        s[9] = min(snapshot.cost_so_far / snapshot.cost_budget, 1.0)
        s[10] = max(0.0, (snapshot.cost_budget - snapshot.cost_so_far) / snapshot.cost_budget)

    s[11] = snapshot.goal_drift_score if snapshot.goal_drift_score is not None else 0.0

    return s


def _build_action_history(snapshot) -> np.ndarray:
    h = np.zeros((HISTORY_LEN, ACTION_COUNT), dtype=np.float32)
    recent_actions = snapshot.recent_actions[-HISTORY_LEN:]
    offset = HISTORY_LEN - len(recent_actions)
    for i, action in enumerate(recent_actions):
        h[offset + i, ActionSpace.encode(action)] = 1.0
    return h.flatten()


def vectorize(snapshot) -> np.ndarray:
    structural = _build_structural(snapshot)
    history = _build_action_history(snapshot)
    vec = np.concatenate([structural, history]).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)
