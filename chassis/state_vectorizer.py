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
  [2]  is_sensitive_tool (0/1)                         - most recent call is SENSITIVE tier
  [3]  consecutive_repeat_count, normalized             - the loop-killer signal
  [4]  recent success_rate (last up to _RECENT_WINDOW tool calls)
  [5]  recent timeout_rate
  [6]  recent error_rate (failed, not timed out)
  [7]  consecutive_error_count, normalized             - trailing errors specifically
  [8]  is_destructive_tool (0/1)                       - most recent call is DESTRUCTIVE tier
  [9]  cost_so_far / cost_budget, capped at 1.0
  [10] remaining_budget_fraction
  [11] goal_drift_score, or 0.0 if not computed for this snapshot

[_STRUCT_DIM : STATE_DIM] one-hot recent-action history, HISTORY_LEN steps
x ACTION_COUNT actions, oldest first, flattened - same shape as worker-prod's
own action-history block, resized for this vocabulary's ACTION_COUNT.
"""
from __future__ import annotations

import re

import numpy as np

from .action_space import ActionSpace

ACTION_COUNT: int = ActionSpace.N
HISTORY_LEN: int = 16
_STRUCT_DIM: int = 12
STATE_DIM: int = _STRUCT_DIM + HISTORY_LEN * ACTION_COUNT
FEATURE_SCHEMA = "xybernetex.state.v1"

_RECENT_WINDOW: int = 8
_MAX_CONSECUTIVE_REPEAT: float = 5.0
_MAX_CONSECUTIVE_ERRORS: float = 5.0

# Canonical source for "is this tool call risky" - teacher/heuristic_teacher.py
# imports these rather than redefining its own copy, the same way worker-prod's
# rollout_generator.py imports keyword helpers from its own rl_chassis.py.
#
# Two tiers, not one boolean: DESTRUCTIVE actions are irreversible enough
# that the teacher should refuse them outright (BLOCK_ACTION); SENSITIVE
# actions have a real external effect (money moves, an email goes out, a
# release happens) but aren't inherently wrong the way a stray `rm` is, so
# the teacher asks a human instead (REQUEST_USER) rather than blocking.
# The state vector carries each tier as its own bit ([2] and [8]). An earlier
# version collapsed them into one "is risky" bit on the theory that the tier
# split was only the teacher's concern - but behavioral cloning can only
# imitate distinctions its input can see: with one bit, the trained policy
# got sensitive-tool REQUEST_USER right on 2/76 cases, predicting
# BLOCK_ACTION for the rest. ([2] used to be an is-repeat bit, which was
# fully derivable from [3].)
#
# Keywords are matched as whole name TOKENS (split on separators and
# camelCase), never as raw substrings - substring matching misclassified
# plenty of benign tools: "eval" in "retrieval", "rm" in "format"/"confirm"/
# "platform", "drop" in "dropdown", "post" in "postgres". Multi-word keywords
# ("run_command", "send_email") match as a contiguous token run.
_DESTRUCTIVE_TOOL_KEYWORDS: tuple[tuple[str, ...], ...] = (
    ("exec",), ("shell",), ("bash",), ("powershell",), ("eval",), ("run", "command"),
    ("delete",), ("remove",), ("rm",), ("drop",), ("truncate",), ("purge",),
)
_SENSITIVE_TOOL_KEYWORDS: tuple[tuple[str, ...], ...] = (
    ("payment",), ("charge",), ("transfer",), ("withdraw",),
    ("send", "email"), ("post",), ("publish",), ("deploy",),
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN_SEPARATOR = re.compile(r"[^a-z0-9]+")


def _name_tokens(tool_name: str) -> list[str]:
    spaced = _CAMEL_BOUNDARY.sub("_", tool_name).lower()
    return [t for t in _TOKEN_SEPARATOR.split(spaced) if t]


def _contains_run(tokens: list[str], keyword: tuple[str, ...]) -> bool:
    n = len(keyword)
    return any(tuple(tokens[i:i + n]) == keyword for i in range(len(tokens) - n + 1))


def risk_tier(tool_name: str) -> str | None:
    """Returns "destructive", "sensitive", or None. Checks destructive
    first - a name matching both (unlikely, but e.g. "delete_and_publish")
    should be treated as the more severe tier."""
    tokens = _name_tokens(tool_name)
    if any(_contains_run(tokens, kw) for kw in _DESTRUCTIVE_TOOL_KEYWORDS):
        return "destructive"
    if any(_contains_run(tokens, kw) for kw in _SENSITIVE_TOOL_KEYWORDS):
        return "sensitive"
    return None


RISK_LABELS: tuple[str, ...] = ("destructive", "sensitive", "none")


def effective_risk_tier(call) -> str | None:
    """The call's risk tier: its own `risk` label when the recorder assessed
    one (ToolCallRecord.risk - e.g. the OpenClaw plugin reading a shell
    command's text), otherwise risk_tier() on the tool name. A name alone
    can't separate `rm -rf` from `python test.py` when both go through one
    shell tool - real OpenClaw logs had every `exec` call judged destructive -
    so a recorded label always wins, including an explicit "none".
    Shared with teacher/heuristic_teacher.py."""
    risk = getattr(call, "risk", None)
    if risk is None:
        return risk_tier(call.tool_name)
    if risk not in RISK_LABELS:
        raise ValueError(f"Unknown risk label {risk!r}; expected one of {RISK_LABELS}")
    return None if risk == "none" else risk


def _same_call(a, b) -> bool:
    return a.tool_name == b.tool_name and a.params == b.params


def consecutive_repeat_count(tool_history: list) -> int:
    """How many calls immediately before the last one are identical to it -
    0 means the last call is not a repeat of anything. Shared with
    teacher/heuristic_teacher.py's loop-detection rule."""
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


def consecutive_error_count(tool_history: list) -> int:
    """How many calls, counting back from the most recent, failed in a row.
    Shared with teacher/heuristic_teacher.py's error-escalation rule."""
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
    tier = effective_risk_tier(history[-1]) if history else None
    s[2] = 1.0 if tier == "sensitive" else 0.0
    s[3] = min(consecutive_repeat_count(history) / _MAX_CONSECUTIVE_REPEAT, 1.0)

    recent = history[-_RECENT_WINDOW:]
    if recent:
        s[4] = sum(1 for c in recent if c.success) / len(recent)
        s[5] = sum(1 for c in recent if c.timed_out) / len(recent)
        s[6] = sum(1 for c in recent if not c.success and not c.timed_out) / len(recent)
    s[7] = min(consecutive_error_count(history) / _MAX_CONSECUTIVE_ERRORS, 1.0)

    s[8] = 1.0 if tier == "destructive" else 0.0

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
    if getattr(snapshot, "feature_schema", FEATURE_SCHEMA) != FEATURE_SCHEMA:
        raise ValueError("v1 vectorizer requires xybernetex.state.v1")
    if getattr(snapshot, "proposed_call", None) is not None:
        raise ValueError("pending proposals require the v2 vectorizer")
    structural = _build_structural(snapshot)
    history = _build_action_history(snapshot)
    vec = np.concatenate([structural, history]).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)
