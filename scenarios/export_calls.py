"""Export each run's user prompt and actual tool calls (raw arguments, and
whether each failed) from OpenClaw's session store, joined with its
scenarios/outcomes.py grade - the input xybernetex-openclaw's
scripts/authz_eval.mjs replays through the real src/authz.js.

The plugin log only ever holds hashed params, so this is the one place raw
commands are recoverable. The output contains them: it's written into the
batch folder (gitignored) and never leaves the machine.

Usage:
    python -m scenarios.export_calls --batch-dir scenarios/batches/NAME
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from compression import zstd
from datetime import datetime
from pathlib import Path

from scenarios.outcome_report import latest_jobs
from scenarios.outcomes import SESSION_DB


def _events(conn: sqlite3.Connection, session_key: str) -> list[dict]:
    events = []
    for (sid,) in conn.execute("select session_id from session_windows where session_key = ?", (session_key,)):
        for raw, packed in conn.execute(
                "select event_json, event_zstd from transcript_events where session_id = ? order by seq", (sid,)):
            text = raw if raw is not None else zstd.decompress(packed).decode("utf-8") if packed else None
            if text:
                events.append(json.loads(text))
    return events


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text")


def run_calls(conn: sqlite3.Connection, job: dict) -> dict:
    """The latest attempt only: session keys are reused across retries."""
    since = datetime.fromisoformat(job["started"]) if job.get("started") else None
    prompt, calls, results = None, [], {}
    for e in _events(conn, job["session_key"]):
        msg = e.get("message") if e.get("type") == "message" else None
        if not msg or (since and e.get("timestamp") and datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")) < since):
            continue
        if msg.get("role") == "user" and prompt is None:
            prompt = _text(msg.get("content"))
        elif msg.get("role") == "assistant":
            calls += [{"id": c.get("id"), "toolName": c.get("name"), "params": c.get("arguments") or {}}
                      for c in msg.get("content") or [] if isinstance(c, dict) and c.get("type") == "toolCall"]
        elif msg.get("role") == "toolResult":
            results[msg.get("toolCallId")] = bool(msg.get("isError"))
    for call in calls:
        call["failed"] = results.get(call.pop("id"), False)
    return {"prompt": prompt, "calls": calls}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=SESSION_DB)
    args = parser.parse_args()
    outcomes = {}
    path = args.batch_dir / "outcomes.jsonl"
    if path.exists():
        outcomes = {r["session_key"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        runs = []
        for job in latest_jobs(args.batch_dir):
            if job.get("status") != "done":
                continue
            grade = outcomes.get(job["session_key"], {})
            runs.append({k: job[k] for k in ("session_key", "scenario", "category", "model")} |
                        {"task": grade.get("task"), "harm": grade.get("harm")} | run_calls(conn, job))
    finally:
        conn.close()
    out = args.batch_dir / "calls.json"
    out.write_text(json.dumps(runs), encoding="utf-8")
    print(f"{len(runs)} runs, {sum(len(r['calls']) for r in runs)} tool calls, "
          f"{sum(r['prompt'] is None for r in runs)} without a prompt -> {out}")


if __name__ == "__main__":
    main()
