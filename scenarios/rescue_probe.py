"""Re-run, once, every job that died (status failed/timeout: the model gave no
usable response) in the given batches - each in a fresh session so the retry
doesn't see the failed attempt - then grade the retries. The share that
succeed is what a retry-on-death wrapper could recover.

Usage:
    python -m scenarios.rescue_probe --batch-dir scenarios/batches/A [--batch-dir ...] --out rescue-probe
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scenarios.outcome_report import grade_batch, latest_jobs
from scenarios.run_suite import AGENT_ID, BATCH_DIR, _openclaw_cmd, _run_job
from scenarios.suite import by_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", default="rescue-probe")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    scenarios, oc = by_id(), _openclaw_cmd()
    out_dir = BATCH_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batch.json").write_text(json.dumps({"batch": args.out, "models": "all", "reps": 1, "only": None,
                                                    "category": None}), encoding="utf-8")
    jobs = []
    for batch_dir in args.batch_dir:
        for j in latest_jobs(batch_dir):
            if j.get("status") in ("failed", "timeout"):
                origin = batch_dir.name
                jobs.append({"batch": args.out, "scenario": j["scenario"], "category": j["category"],
                             "model": j["model"], "rep": j["rep"], "origin": origin,
                             "session_key": f"agent:{AGENT_ID}:scn-{args.out}-{origin}-{j['scenario']}-{j['model']}-r{j['rep']}".lower(),
                             "folder": f"runs/{args.out}/{origin}/{j['scenario']}/{j['model']}-r{j['rep']}"})
    print(f"retrying {len(jobs)} dead runs once each", flush=True)
    manifest = out_dir / "runs.jsonl"
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_run_job, j, scenarios[j["scenario"]], oc, args.timeout) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            result = f.result()
            with open(manifest, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(result) + "\n")
            print(f"[{i}/{len(jobs)}] {result['status']:7s} {result['scenario']:18s} {result['model']}", flush=True)

    graded = grade_batch(out_dir)
    alive = [r for r in graded if not r["detail"].startswith("run ")]
    won = [r for r in alive if r["task"] is True]
    print(f"\nretries: {len(graded)}; died again: {len(graded) - len(alive)}; finished: {len(alive)}; "
          f"completed the task: {len(won)} ({len(won) / max(len(graded), 1):.0%} of dead runs rescued)")


if __name__ == "__main__":
    main()
