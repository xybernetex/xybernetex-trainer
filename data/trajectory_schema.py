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
