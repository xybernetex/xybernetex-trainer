"""Shared snapshot format for one point in an OpenClaw-supervised trajectory.

This is a first draft - expected to grow/change as chassis/state_vectorizer.py
and teacher/heuristic_teacher.py get built out and reveal what's actually
needed. Every other module (synthetic_generator, replay_loader,
state_vectorizer, heuristic_teacher) should agree on this shape rather than
each inventing its own, the same role worker-prod's SQLite WorkingMemory
schema plays there - except this is a plain in-memory dataclass, not a
database, since the supervisor's state is whatever the Durable Object
already tracks per session, not something this trainer needs to query live.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCallRecord:
    tool_name: str
    params: dict
    success: bool
    error: str | None = None
    timed_out: bool = False
    # Risk tier assessed by whoever recorded the call, from what it actually
    # did - "destructive", "sensitive", or "none" - e.g. the OpenClaw plugin
    # classifies a shell tool's command text on the user's machine, where the
    # text itself never leaves. None means "not assessed": the vectorizer then
    # falls back to judging the tool NAME (state_vectorizer.effective_risk_tier),
    # which is all synthetic data carries.
    risk: str | None = None


@dataclass
class ProposedToolCall:
    """A pending operation has no success, timeout or other execution outcome."""
    tool_name: str
    params: dict
    risk: str | None = None
    tool_call_id: str | None = None


@dataclass
class TrajectorySnapshot:
    run_id: str
    objective: str
    step_number: int
    max_steps: int
    recent_actions: list[str] = field(default_factory=list)   # supervisor's own past actions, oldest first
    tool_history: list[ToolCallRecord] = field(default_factory=list)
    cost_so_far: float = 0.0
    cost_budget: float = 0.0
    # Cosine distance between an embedding of recent tool-call context and
    # the stated objective - precomputed by whoever builds this snapshot
    # (the live serving path, synthetic_generator, or replay_loader), never
    # by state_vectorizer.py itself. That keeps the vectorizer's "zero
    # network calls" purity guarantee intact - an embedding call is exactly
    # the kind of I/O chassis/__init__.py's docstring rules out. None means
    # "not computed for this snapshot" (e.g. early synthetic data before
    # this feature existed), not "zero drift" - the vectorizer must treat
    # those differently.
    goal_drift_score: float | None = None
    feature_schema: str = "xybernetex.state.v1"
    proposed_call: ProposedToolCall | None = None
