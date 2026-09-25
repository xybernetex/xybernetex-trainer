"""Generate alpha v2 parity fixtures independently of the deployed v1 cases."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from chassis.proposal_vectorizer import FEATURE_SCHEMA, STATE_DIM, EXTRA_FEATURES, vectorize_proposal
from data.trajectory_schema import ProposedToolCall, ToolCallRecord, TrajectorySnapshot


def cases():
    for size in (0, 1, 3, 8, 12):
        for mode in ("identical", "varied"):
            history = [ToolCallRecord("read" if mode == "identical" else f"tool{i % 3}", {"h": "abc"},
                                      success=i % 3 != 0) for i in range(size)]
            for name, risk in [("read", "none"), ("exec", "destructive"), ("message", "sensitive"), ("mystery", None)]:
                for present in (False, True):
                    snapshot = TrajectorySnapshot("fixture", "", size, 50, tool_history=history,
                        recent_actions=["CONTINUE", "INJECT_CONTEXT"] if size else [],
                        feature_schema=FEATURE_SCHEMA,
                        goal_drift_score=0.0 if mode == "identical" else None,
                        proposed_call=ProposedToolCall(name, {"h": "abc"}, risk, "call") if present else None)
                    yield f"{size}-{mode}-{name}-{present}", snapshot
    for left, right in [({"a": 1, "b": [2]}, {"b": [2], "a": 1}), ({"a": True}, {"a": 1}),
                        ({"a": None}, {"a": 0}), ({"a": []}, {"a": {}})]:
        yield f"equality-{left}-{right}", TrajectorySnapshot("eq", "", 1, 50,
            tool_history=[ToolCallRecord("read", left, True)], feature_schema=FEATURE_SCHEMA,
            proposed_call=ProposedToolCall("read", right, "none"))


def build_cases():
    return {"feature_schema": FEATURE_SCHEMA, "state_dim": STATE_DIM, "extra_features": list(EXTRA_FEATURES),
            "cases": [{"name": name, "snapshot": asdict(s), "features": vectorize_proposal(s).tolist()}
                      for name, s in cases()]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_cases()
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(payload['cases'])} proposal cases")
