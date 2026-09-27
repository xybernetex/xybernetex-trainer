"""Parses the xybernetex-openclaw plugin's observe-only log back into
TrajectorySnapshot sequences.

The plugin (xybernetex-openclaw, src/supervisor.js) appends one JSON line per
completed tool call to ~/.openclaw/xybernetex-supervisor.jsonl, holding the
exact snapshot it sent the policy and the policy's answer, plus one
`runUsage` line per run with its token total. Snapshots are rebuilt as-is -
params arrive as {"h": <hash>}, which is all the vectorizer and teacher need
(they only compare params for equality) - so a replayed snapshot vectorizes
to exactly what the Worker saw.

Lines this doesn't recognize (disabled-plugin notices, old debug lines) are
skipped rather than failing the whole load.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from data.trajectory_schema import ToolCallRecord, TrajectorySnapshot, ProposedToolCall

DEFAULT_LOG_PATH = Path.home() / ".openclaw" / "xybernetex-supervisor.jsonl"


@dataclass
class ReplayStep:
    snapshot: TrajectorySnapshot
    tool_name: str
    ts: str
    policy_action: str | None       # None when the policy call failed
    policy_probs: dict | None
    error: str | None


@dataclass
class ReplayRun:
    run_key: str
    session_key: str | None = None
    steps: list[ReplayStep] = field(default_factory=list)
    usage: dict | None = None
    model: str | None = None


def _snapshot_from_json(raw: dict) -> TrajectorySnapshot:
    history = [
        ToolCallRecord(
            tool_name=call["tool_name"],
            params=call.get("params", {}),
            success=call["success"],
            error=call.get("error"),
            timed_out=call.get("timed_out", False),
            risk=call.get("risk"),
            authorization=call.get("authorization"),
        )
        for call in raw.get("tool_history", [])
    ]
    return TrajectorySnapshot(
        run_id=raw.get("run_id", ""),
        objective=raw.get("objective", ""),
        step_number=raw["step_number"],
        max_steps=raw["max_steps"],
        recent_actions=list(raw.get("recent_actions", [])),
        tool_history=history,
        cost_so_far=raw.get("cost_so_far", 0.0),
        cost_budget=raw.get("cost_budget", 0.0),
        goal_drift_score=raw.get("goal_drift_score"),
        feature_schema=raw.get("feature_schema", "xybernetex.state.v1"),
        proposed_call=ProposedToolCall(**raw["proposed_call"]) if raw.get("proposed_call") is not None else None,
    )


def load_runs(log_path: str | Path = DEFAULT_LOG_PATH, since: str | None = None) -> dict[str, ReplayRun]:
    """run_key -> ReplayRun, steps in step order. `since` filters out lines
    whose ISO timestamp sorts before it."""
    runs: dict[str, ReplayRun] = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict) or "debug" in entry or "runKey" not in entry or entry.get("type") == "tool_proposal":
                continue
            if since and entry.get("ts", "") < since:
                continue
            run = runs.setdefault(entry["runKey"], ReplayRun(run_key=entry["runKey"]))
            if entry.get("sessionKey"):
                run.session_key = entry["sessionKey"]
            if "runUsage" in entry:
                run.usage = entry["runUsage"]
                run.model = entry.get("model")
                continue
            if "snapshot" not in entry:
                continue
            decision = entry.get("decision") or {}
            run.steps.append(ReplayStep(
                snapshot=_snapshot_from_json(entry["snapshot"]),
                tool_name=entry.get("toolName", ""),
                ts=entry.get("ts", ""),
                policy_action=decision.get("action"),
                policy_probs=decision.get("action_probs"),
                error=entry.get("error"),
            ))
    for run in runs.values():
        run.steps.sort(key=lambda s: s.snapshot.step_number)
    return runs


def load_sessions(log_dir: str):
    """The original placeholder interface: every run in the log under log_dir."""
    return list(load_runs(Path(log_dir) / DEFAULT_LOG_PATH.name).values())


def load_proposals(log_path: str | Path = DEFAULT_LOG_PATH) -> list[TrajectorySnapshot]:
    """Load pending-call telemetry without treating proposals as executed steps."""
    snapshots = []
    with open(log_path, encoding="utf-8") as stream:
        for line in stream:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and entry.get("type") == "tool_proposal":
                snapshot = _snapshot_from_json(entry["snapshot"])
                if snapshot.feature_schema != "xybernetex.state.v2" or snapshot.proposed_call is None:
                    raise ValueError("tool_proposal requires a v2 snapshot with proposed_call")
                snapshots.append(snapshot)
    return snapshots
