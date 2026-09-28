"""Writes (snapshot JSON, expected feature vector) pairs from the REAL
chassis/state_vectorizer.py, for xybernetex-cfworker's JS port of it
(src/state_vectorizer.js) to match bit-for-bit.

The Worker builds features itself from a raw TrajectorySnapshot, so the
Python vectorizer the policy was trained on and the JS one it's served with
must agree exactly - a silent mismatch doesn't error, it just feeds the
policy inputs it never saw in training. This file is how that agreement is
checked: rerun it whenever state_vectorizer.py changes and commit the output
into xybernetex-cfworker (its test suite fails until the JS side catches up).

Cases are the hand-built edge cases below (each aimed at a specific way a
port could drift from Python semantics) plus every step of one seeded
synthetic episode per category.

Usage:
    python -m export.export_vectorizer_cases --output ../xybernetex-cfworker/test/fixtures/vectorizer_cases.json
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from chassis.action_space import ActionSpace
from chassis.state_vectorizer import STATE_DIM, STATE_DIM_V3, vectorize, vectorize_v3
from data.synthetic_generator import _CATEGORIES, episode_snapshots
from data.trajectory_schema import ToolCallRecord, TrajectorySnapshot

SEED = 7


def _snap(**overrides) -> TrajectorySnapshot:
    fields = dict(run_id="case", objective="case", step_number=3, max_steps=20,
                  cost_so_far=1.0, cost_budget=10.0)
    fields.update(overrides)
    return TrajectorySnapshot(**fields)


def _call(tool_name="read_file", params=None, success=True, timed_out=False, risk=None,
          authorization=None) -> ToolCallRecord:
    return ToolCallRecord(tool_name=tool_name, params={"arg": 1} if params is None else params,
                          success=success, timed_out=timed_out, risk=risk, authorization=authorization)


def edge_cases() -> list[tuple[str, TrajectorySnapshot]]:
    all_actions = list(ActionSpace.PRIMITIVES)
    return [
        ("empty history", _snap()),
        ("max_steps zero", _snap(step_number=0, max_steps=0)),
        ("step past max", _snap(step_number=25, max_steps=20)),
        ("zero cost budget", _snap(cost_so_far=5.0, cost_budget=0.0)),
        ("cost over budget", _snap(cost_so_far=12.5, cost_budget=10.0)),
        ("goal drift set", _snap(goal_drift_score=0.37)),
        ("goal drift float32 rounding", _snap(goal_drift_score=0.1)),
        ("float32 overflow -> 1.0", _snap(step_number=10**40, max_steps=1, goal_drift_score=1e39)),
        ("float32 negative overflow -> 0.0", _snap(goal_drift_score=-1e39)),
        ("huge cost ratio", _snap(cost_so_far=1e300, cost_budget=1e-300)),
        ("more than 16 past actions", _snap(recent_actions=(all_actions * 3)[:21])),
        ("destructive snake_case", _snap(tool_history=[_call("drop_table")])),
        ("destructive camelCase", _snap(tool_history=[_call("runCommand")])),
        ("sensitive camelCase", _snap(tool_history=[_call("sendEmail")])),
        ("acronym camelCase", _snap(tool_history=[_call("HTTPDelete")])),
        ("both tiers -> destructive", _snap(tool_history=[_call("delete_and_publish")])),
        ("substring is not a token", _snap(tool_history=[_call("google_drive_file_retrieval")])),
        ("postgres is not post", _snap(tool_history=[_call("postgres_query")])),
        ("send without email", _snap(tool_history=[_call("send_message")])),
        ("non-ascii name", _snap(tool_history=[_call("ÜberDelete")])),
        ("recorded risk none overrides a risky name", _snap(tool_history=[_call("exec", risk="none")])),
        ("recorded destructive on a benign name", _snap(tool_history=[_call("read", risk="destructive")])),
        ("recorded sensitive on a shell tool", _snap(tool_history=[_call("exec", risk="sensitive")])),
        ("unlabelled shell tool falls back to its name", _snap(tool_history=[_call("exec")])),
        ("only the last call's label counts",
         _snap(tool_history=[_call("exec", risk="destructive"), _call("exec", risk="none")])),
        ("only the last call's tier counts",
         _snap(tool_history=[_call("delete_file"), _call("read_file")])),
        ("repeat with reordered params",
         _snap(tool_history=[_call(params={"a": 1, "b": [1, {"c": 2}]}),
                             _call(params={"b": [1, {"c": 2}], "a": 1})])),
        ("repeat 1 vs 1.0 vs True (Python ==)",
         _snap(tool_history=[_call(params={"x": 1}), _call(params={"x": 1.0}), _call(params={"x": True})])),
        ("None is not 0", _snap(tool_history=[_call(params={"x": None}), _call(params={"x": 0})])),
        ("list vs dict params", _snap(tool_history=[_call(params={"x": []}), _call(params={"x": {}})])),
        ("same params different tool", _snap(tool_history=[_call("a_tool"), _call("b_tool")])),
        ("repeat count saturates",
         _snap(tool_history=[_call() for _ in range(9)])),
        ("repeat broken in the middle",
         _snap(tool_history=[_call(), _call(), _call(params={"arg": 2}), _call(), _call()])),
        ("errors and timeouts",
         _snap(tool_history=[_call(), _call(success=False, timed_out=True), _call(success=False),
                             _call(success=True, timed_out=True), _call(success=False)])),
        ("error run saturates", _snap(tool_history=[_call(success=False) for _ in range(7)])),
        ("recent window is 8",
         _snap(tool_history=[_call(success=False) for _ in range(5)] + [_call(params={"arg": i}) for i in range(8)])),
        ("authorization requested", _snap(tool_history=[_call("exec", risk="destructive", authorization="requested")])),
        ("authorization own_artifact", _snap(tool_history=[_call("exec", risk="destructive", authorization="own_artifact")])),
        ("authorization unrequested", _snap(tool_history=[_call("exec", risk="sensitive", authorization="unrequested")])),
        ("only the last call's authorization counts",
         _snap(tool_history=[_call("exec", risk="destructive", authorization="unrequested"),
                             _call("exec", risk="destructive", authorization="requested")])),
        ("an earlier label is not carried to an unlabelled last call",
         _snap(tool_history=[_call("exec", risk="destructive", authorization="requested"), _call()])),
        ("v3-tagged snapshot", _snap(feature_schema="xybernetex.state.v3",
                                     tool_history=[_call("exec", risk="destructive", authorization="unrequested")])),
    ]


def synthetic_cases() -> list[tuple[str, TrajectorySnapshot]]:
    rng = random.Random(SEED)
    cases = []
    for category in _CATEGORIES:
        for snapshot, _ in episode_snapshots(category, rng):
            cases.append((f"synthetic {category} step {snapshot.step_number}", snapshot))
    return cases


def build_cases() -> dict:
    # The float32-overflow edge cases overflow on purpose.
    # A v3-tagged snapshot has no v1 vector (the v1 extractor rejects it).
    with np.errstate(over="ignore"):
        cases = [
            {"name": name, "snapshot": asdict(snapshot),
             "features": vectorize(snapshot).tolist() if snapshot.feature_schema == "xybernetex.state.v1" else None,
             "features_v3": vectorize_v3(snapshot).tolist()}
            for name, snapshot in edge_cases() + synthetic_cases()
        ]
    return {"state_dim": STATE_DIM, "state_dim_v3": STATE_DIM_V3, "primitives": list(ActionSpace.PRIMITIVES),
            "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_cases()
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(payload['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
