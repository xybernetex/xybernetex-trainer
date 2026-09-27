"""Grade whether each run in a scenario batch actually accomplished its task
(scenarios/outcomes.py), and relate that to what the teacher and policy
decided during the run.

Writes <batch-dir>/outcomes.jsonl (one line per job) and a Markdown summary.
Observe-only data means every relationship reported here is correlational:
an intervention that "precedes failure" wasn't enforced, so it didn't cause
anything - it only marks states from which the agent tended to fail, which is
exactly the target a value function learns.

Usage:
    python -m scenarios.outcome_report --batch-dir scenarios/batches/NAME --output out/NAME-outcomes.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data.replay_loader import DEFAULT_LOG_PATH
from scenarios.outcomes import evaluate, final_replies
from scenarios.report import build_report
from scenarios.run_suite import WORKSPACE
from scenarios.suite import by_id


def latest_jobs(batch_dir: Path) -> list[dict]:
    latest: dict[str, dict] = {}
    for line in (batch_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            job = json.loads(line)
            latest[job["session_key"]] = job
    return list(latest.values())


def grade_batch(batch_dir: Path, workers: int = 8) -> list[dict]:
    jobs = latest_jobs(batch_dir)
    done = [j for j in jobs if j.get("status") == "done"]
    stored = {j["session_key"]: j.get("reply") for j in done}
    looked_up = final_replies([j for j in done if not stored[j["session_key"]]])
    scenarios = by_id()

    def grade(job: dict) -> dict:
        base = {k: job[k] for k in ("session_key", "scenario", "category", "model", "rep")}
        if job.get("status") != "done":
            return {**base, "task": None, "harm": None, "detail": f"run {job.get('status')}", "reply_found": False}
        reply = stored[job["session_key"]] or looked_up.get(job["session_key"])
        out = evaluate(scenarios[job["scenario"]], WORKSPACE / job["folder"], reply)
        return {**base, "task": out.task, "harm": out.harm, "detail": out.detail, "reply_found": bool(reply)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(grade, jobs))


def _rate(rows: list[dict]) -> str:
    known = [r for r in rows if r["task"] is not None]
    if not known:
        return "n/a"
    ok = sum(r["task"] for r in known)
    return f"{ok / len(known):.0%} ({ok}/{len(known)})"


def summarize(batch: str, outcomes: list[dict], decisions: dict[str, dict]) -> str:
    lines = [f"# Task outcomes: {batch}", "",
             "Whether each run accomplished what it was asked, from its run folder and final reply. "
             "`n/a` = unverifiable or unachievable by design. Every supervisor relationship below is "
             "correlational: nothing was enforced during these runs.", "",
             f"Runs: {len(outcomes)}. Verifiable: {sum(r['task'] is not None for r in outcomes)}. "
             f"Task success: {_rate(outcomes)}. Planted-instruction harm: "
             f"{sum(bool(r['harm']) for r in outcomes)}/{sum(r['harm'] is not None for r in outcomes)} injection runs.",
             "", "## By category", "", "| Category | Runs | Task success |", "|---|---:|---|"]
    for cat, rows in sorted(_group(outcomes, "category").items()):
        lines.append(f"| {cat} | {len(rows)} | {_rate(rows)} |")
    lines += ["", "## By agent model", "", "| Model | Task success |", "|---|---|"]
    for model, rows in sorted(_group(outcomes, "model").items()):
        lines.append(f"| {model} | {_rate(rows)} |")

    lines += ["", "## Supervisor decisions vs. outcome", "",
              "Task success among verifiable runs, split by whether a given action was chosen at least once "
              "during the run.", "", "| Action | Chooser | Runs where chosen | Success when chosen | "
              "Success when not |", "|---|---|---:|---|---|"]
    verifiable = [r for r in outcomes if r["task"] is not None and r["session_key"] in decisions]
    for action in ("REPLAN", "BLOCK_ACTION", "INJECT_CONTEXT", "REQUEST_USER", "TERMINATE"):
        for who, key in (("teacher", "teacher_actions"), ("policy", "actions")):
            chosen = [r for r in verifiable if decisions[r["session_key"]][key].get(action)]
            if chosen:
                rest = [r for r in verifiable if r not in chosen]
                lines.append(f"| {action} | {who} | {len(chosen)} | {_rate(chosen)} | {_rate(rest)} |")

    lines += ["", "## Failed runs", "", "| Scenario | Model | Why |", "|---|---|---|"]
    for r in sorted((r for r in outcomes if r["task"] is False), key=lambda r: (r["scenario"], r["model"])):
        lines.append(f"| {r['scenario']} | {r['model']} | {r['detail']} |")
    return "\n".join(lines) + "\n"


def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--output", type=Path, help="Markdown summary path")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    outcomes = grade_batch(args.batch_dir, args.workers)
    with open(args.batch_dir / "outcomes.jsonl", "w", encoding="utf-8") as f:
        for row in outcomes:
            f.write(json.dumps(row) + "\n")
    report = build_report(args.batch_dir, args.log)
    decisions = {row["session_key"]: row for row in report["rows"]}
    markdown = summarize(report["batch"], outcomes, decisions)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"final replies found for {sum(r['reply_found'] for r in outcomes)} of {len(outcomes)} runs")


if __name__ == "__main__":
    main()
