"""Audit pending telemetry and export an unlabelled review set, without inference.

Local rule decisions are evidence of configured policy, not general ground truth.
Missing outcomes never imply success. Join only by run and host call ID.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from chassis.proposal_vectorizer import vectorize_proposal
from data.replay_loader import _snapshot_from_json


def audit(entries):
    groups = defaultdict(list)
    for index, entry in enumerate(entries):
        if entry.get("runKey") and entry.get("toolCallId"):
            groups[entry["runKey"], entry["toolCallId"]].append((index, entry))
    rows = []
    for index, entry in enumerate(entries):
        if entry.get("type") != "tool_proposal":
            continue
        snapshot = entry["snapshot"]
        features = vectorize_proposal(_snapshot_from_json(snapshot))
        proposal = snapshot["proposed_call"]
        key = entry.get("runKey"), entry.get("toolCallId")
        related = groups[key] if all(key) else []
        proposals = [(i, e) for i, e in related if e.get("type") == "tool_proposal"]
        completed = [(i, e) for i, e in related if e.get("type") == "tool_completed"]
        gates = [(i, e) for i, e in related if e.get("type") == "tool_gate"]
        problems = []
        if snapshot.get("run_id") != key[0] or proposal.get("tool_call_id") != key[1]:
            problems.append("inconsistent_snapshot_identity")
        if not all(key):
            problems.append("missing_correlation_id")
        elif len(proposals) != 1 or len(completed) > 1 or len(gates) > 1:
            problems.append("ambiguous_correlation")
        for other_index, other in completed + gates:
            if other_index <= index:
                problems.append("outcome_precedes_proposal")
            name = other.get("record", {}).get("tool_name", other.get("toolName"))
            digest = other.get("record", {}).get("params", {}).get("h", other.get("paramsHash"))
            if name != proposal["tool_name"] or digest != proposal["params"].get("h"):
                problems.append("call_changed_or_mismatched")
        blocked = any(e.get("enforced") and e.get("action") == "BLOCK_ACTION" for _, e in gates)
        if blocked and any(e["record"].get("success") for _, e in completed):
            problems.append("blocked_call_reported_success")
        status = "invalid" if problems else "blocked" if blocked else "completed" if completed else "pending_or_missing"
        rows.append({"run_id": key[0], "tool_call_id": key[1], "status": status,
                     "problems": sorted(set(problems)), "snapshot": snapshot,
                     "feature_count": len(features),
                     "completed": [e["record"] for _, e in completed],
                     "configured_rule_actions": [{"rule_id": e.get("ruleId"), "action": e.get("action"),
                                                  "enforced": e.get("enforced")} for _, e in gates],
                     "target_action": None, "review_status": "unreviewed"})
    unmatched = sum(not any(other.get("type") == "tool_proposal" for _, other in
                           groups.get((entry.get("runKey"), entry.get("toolCallId")), []))
                    for entry in entries if entry.get("type") == "tool_completed")
    return {"proposals": len(rows), "unmatched_completions": unmatched,
            "statuses": dict(Counter(r["status"] for r in rows)),
            "labelled": 0, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = []
    for path in args.logs:
        entries.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    report = audit(entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
    if report["statuses"].get("invalid") or report["unmatched_completions"] or not report["proposals"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
