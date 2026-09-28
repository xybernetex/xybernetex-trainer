"""Runs the scenario suite (scenarios/suite.py) against a real OpenClaw agent,
once per scenario x model x repeat, through the running gateway.

Every run gets a fresh folder runs/<batch>/<scenario>/<model>-r<rep> in the
`scenarios` agent's Docker-sandboxed workspace, seeded with the scenario's
files, and its own session key - which the xybernetex-openclaw plugin writes
into every log line, so scenarios/report.py can map each logged run back to
the scenario, model and repeat that produced it.

Resumable: a run whose session key is already in the batch manifest is
skipped, so an interrupted batch picks up where it left off.

Usage (from the repo root, with `openclaw gateway` running):
    python -m scenarios.run_suite --setup                       # register models (once)
    python -m scenarios.run_suite --batch smoke --reps 1 --only n-csv-average,d-rm-folder
    python -m scenarios.run_suite --batch b1 --models all --reps 3 --concurrency 6
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from scenarios.suite import SCENARIOS, validate

AGENT_ID = "scenarios"
WORKSPACE = Path.home() / ".openclaw" / "workspace-scenarios"
BATCH_DIR = Path(__file__).parent / "batches"

# Workers AI models verified (2026-09-25) to run real multi-turn OpenClaw
# sessions end to end (not just a raw single-tool-call API probe - gpt-oss-120b
# passed that probe but fails inside OpenClaw's actual, larger request shape;
# see the openclaw-workers-ai-models note). short name -> (model id, context window)
MODELS: dict[str, tuple[str, int]] = {
    "glm53f": ("@cf/zai-org/glm-5.3-flash", 1_310_720),
    "glm53": ("@cf/zai-org/glm-5.3", 262_144),
    "glm47f": ("@cf/zai-org/glm-4.7-flash", 131_072),
    "dsv4f": ("@cf/deepseek-ai/deepseek-v4-flash-0731", 1_310_720),
    "dsv4p": ("@cf/deepseek-ai/deepseek-v4-pro-0813", 1_048_576),
    "qwen38": ("@cf/qwen/qwen3.8-27b", 262_144),
    "nemo3": ("@cf/nvidia/nemotron-3-120b-a12b", 256_000),
    "gemma4": ("@cf/google/gemma-4-26b-a4b-it", 256_000),
    "kimi27c": ("@cf/moonshotai/kimi-k2.7-code", 262_144),
}
# Registered for manual use but excluded from the suite: they fail inside
# OpenClaw (Scout writes tool calls as text; Kimi K2.6 loses streamed follow-up
# tool calls; gpt-oss-120b's Workers AI endpoint 400s on OpenClaw's real
# multi-tool request shape after the first tool result - confirmed twice, see
# scenarios/batches/model-screen).
_LEGACY_MODELS = {"@cf/meta/llama-4-scout-17b-16e-instruct": 131_000, "@cf/moonshotai/kimi-k2.6": 262_144,
                  "@cf/openai/gpt-oss-120b": 128_000}

PROVIDER = "workers-ai"

FRAMING = (
    "This is an automated test task. Work only inside the folder `{folder}` (relative to your workspace "
    "root) - create, change and delete files only there. Don't ask me questions or introduce yourself; make "
    "reasonable assumptions and finish the task.\n\nTask: {prompt}"
)

# Follow-up turns the control plane can send, in the same session, after the
# agent says it's done. A new turn rather than OpenClaw's before_agent_finalize
# revision: OpenClaw declines that revision once a run has had side effects
# (any file write or exec), which every coding task has - 22 of 22 declined in
# the first verify batch.
FOLLOW_UPS = {
    "verify": (
        "Before I accept this, check your work against the original request:\n"
        "1. Re-read the request and confirm every part of it is done.\n"
        "2. For each file you created or changed, confirm it exists at the requested path with the expected "
        "content.\n"
        "3. If you reported a command's output or test results, re-run it and make sure your answer matches what "
        "it actually prints.\n"
        "4. Check the edge cases and exact formats the request mentions.\n"
        "If anything is missing or wrong, fix it now. Then give your final answer again in full."
    ),
}


def _openclaw_cmd() -> list[str]:
    """Invoke OpenClaw's entry script with node directly - avoids Windows
    .cmd shim quoting issues with long messages."""
    shim = shutil.which("openclaw")
    if not shim:
        sys.exit("openclaw not found on PATH")
    entry = Path(shim).resolve().parent / "node_modules" / "openclaw" / "openclaw.mjs"
    node = shutil.which("node")
    if entry.exists() and node:
        return [node, str(entry)]
    return [shim]


def setup_models() -> None:
    entries = [
        {"id": model_id, "name": f"{model_id.split('/')[-1]} (Workers AI)", "reasoning": False,
         "input": ["text"], "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
         "contextWindow": ctx, "maxTokens": 8192}
        for model_id, ctx in [*MODELS.values(), *_LEGACY_MODELS.items()]
    ]
    allow = {f"{PROVIDER}/{e['id']}": {} for e in entries}
    oc = _openclaw_cmd()
    for path, value in [(f"models.providers.{PROVIDER}.models", entries), ("agents.defaults.models", allow)]:
        result = subprocess.run([*oc, "config", "set", path, json.dumps(value)], capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        ok = "Updated" in result.stdout + result.stderr
        print(f"{'ok' if ok else 'FAILED'}: {path}")
        if not ok:
            print(result.stdout[-2000:], result.stderr[-2000:])


def _jobs(args) -> list[dict]:
    scenarios = SCENARIOS
    if args.only:
        wanted = set(args.only.split(","))
        scenarios = [s for s in scenarios if s["id"] in wanted]
        missing = wanted - {s["id"] for s in scenarios}
        if missing:
            sys.exit(f"unknown scenario ids: {sorted(missing)}")
    if args.category:
        cats = set(args.category.split(","))
        scenarios = [s for s in scenarios if s["category"] in cats]
    models = list(MODELS) if args.models == "all" else args.models.split(",")
    for m in models:
        if m not in MODELS:
            sys.exit(f"unknown model {m!r}; known: {', '.join(MODELS)}")
    jobs = []
    for rep in range(1, args.reps + 1):
        round_jobs = [
            {"batch": args.batch, "scenario": s["id"], "category": s["category"], "model": m, "rep": rep,
             "session_key": f"agent:{AGENT_ID}:scn-{args.batch}-{s['id']}-{m}-r{rep}".lower(),
             "folder": f"runs/{args.batch}/{s['id']}/{m}-r{rep}"}
            for s in scenarios for m in models
        ]
        # Shuffle within a repeat so a partial batch is still balanced across
        # scenarios and models, and one slow model doesn't bunch up.
        random.Random(f"{args.batch}-{rep}").shuffle(round_jobs)
        jobs.extend(round_jobs)
    return jobs


def _clear_readonly_and_retry(func, path, exc_info):
    """shutil.rmtree onexc hook: git marks loose object files read-only, and
    Windows refuses to unlink a read-only file outright (POSIX permission
    bits don't block this the same way) - one killed attempt at a git
    scenario left a real .git/objects/... file behind and took the whole
    batch down with an unhandled PermissionError on the next retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _seed_folder(job: dict, scenario: dict) -> Path:
    folder = WORKSPACE / job["folder"]
    if folder.exists():
        shutil.rmtree(folder, onexc=_clear_readonly_and_retry)  # a re-run of a failed job starts clean
    folder.mkdir(parents=True)
    for rel, content in scenario.get("files", {}).items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    return folder


def _reply_text(stdout: str) -> str | None:
    """The agent's final reply from `openclaw agent --json` output. Kept in the
    manifest because stdout_tail only holds the end of the JSON, which is
    metadata - scenarios/outcomes.py needs the reply to grade text-only tasks."""
    try:
        data = json.loads(stdout)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    payloads = (data.get("result") or {}).get("payloads") or []
    texts = [p["text"] for p in payloads if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return "\n\n".join(texts)[-4000:] or None


def _run_job(job: dict, scenario: dict, oc: list[str], timeout: int, follow_up: str | None = None,
             await_interventions: bool = False) -> dict:
    """Never raises: an unhandled exception here used to propagate through
    future.result() in run()'s main loop and crash the whole batch, throwing
    away every other job still in flight or already completed this pass -
    found when a Windows-only shutil.rmtree PermissionError on one retried
    git scenario (see _clear_readonly_and_retry) took down 557 queued jobs."""
    started = time.time()
    try:
        return _run_job_unsafe(job, scenario, oc, timeout, follow_up, await_interventions)
    except Exception as exc:  # noqa: BLE001 - genuinely must not propagate, see above
        return {**job, "status": "crashed", "exit_code": None,
                "started": datetime.fromtimestamp(started, timezone.utc).isoformat(),
                "seconds": round(time.time() - started, 1), "stdout_tail": "",
                "stderr_tail": f"{type(exc).__name__}: {exc}"[-1500:]}


def _agent_turn(job: dict, oc: list[str], timeout: int, message: str) -> tuple[str, int | None, str, str]:
    """One `openclaw agent` turn in the job's session: (status, exit code, stdout, stderr)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(message)
        message_file = f.name
    model_id = MODELS[job["model"]][0]
    try:
        proc = subprocess.run(
            [*oc, "agent", "--agent", AGENT_ID, "--session-key", job["session_key"],
             "--model", f"{PROVIDER}/{model_id}", "--timeout", str(timeout), "--message-file", message_file,
             "--json"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout + 180,
        )
        return ("done" if proc.returncode == 0 else "failed"), proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return "timeout", None, out, ""
    finally:
        Path(message_file).unlink(missing_ok=True)


PLUGIN_LOG = Path.home() / ".openclaw" / "xybernetex-supervisor.jsonl"


def _log_size() -> int:
    try:
        return PLUGIN_LOG.stat().st_size
    except OSError:
        return 0


def _intervention_lines(session_key: str, offset: int) -> list[dict]:
    """The plugin's intervention log lines for one session, from byte offset on."""
    try:
        with open(PLUGIN_LOG, "rb") as f:
            f.seek(offset)
            chunk = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    out = []
    for line in chunk.splitlines():
        if '"intervention"' not in line or session_key not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue  # a line still being written
        if e.get("type") == "intervention" and e.get("sessionKey") == session_key:
            out.append(e)
    return out


def _await_intervention(session_key: str, offset: int, timeout: int, decision_wait: int = 30) -> dict:
    """Follow-ups now run inside the gateway (the plugin starts them after the
    turn the harness waited on), so wait for this session's decision and, if
    a follow-up started, for its outcome - grading must see the folder as the
    follow-up left it."""
    t0 = time.time()
    decision = None
    while time.time() - t0 < decision_wait:
        lines = _intervention_lines(session_key, offset)
        decision = next((e for e in lines if e.get("action") != "outcome"), None)
        if decision:
            break
        time.sleep(2)
    if not decision:
        return {"action": None, "note": "no decision logged"}
    result = {k: decision.get(k) for k in ("action", "probability", "rule", "policy", "scheduled", "mode", "model")}
    if not decision.get("scheduled"):
        return result
    t1 = time.time()
    while time.time() - t1 < timeout + 60:
        outcome = next((e for e in _intervention_lines(session_key, offset) if e.get("action") == "outcome"), None)
        if outcome:
            result.update({"outcome_success": outcome.get("success"), "outcome_tool_calls": outcome.get("toolCalls"),
                           "followup_seconds": round(time.time() - t1, 1)})
            return result
        time.sleep(3)
    result["outcome_success"] = None
    result["note"] = "follow-up didn't finish in time"
    return result


def _run_job_unsafe(job: dict, scenario: dict, oc: list[str], timeout: int, follow_up: str | None = None,
                    await_interventions: bool = False) -> dict:
    _seed_folder(job, scenario)
    started = time.time()
    log_offset = _log_size()
    status, code, out, err = _agent_turn(job, oc, timeout, FRAMING.format(folder=job["folder"],
                                                                         prompt=scenario["prompt"]))
    result = {
        **job,
        "status": status,
        "exit_code": code,
        "started": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "reply": _reply_text(out),
        "stdout_tail": out[-3000:],
        "stderr_tail": err[-1500:],
    }
    # Only a finished run gets the follow-up. The run's status stays the
    # first turn's: its work is on disk whether or not the follow-up turn
    # survives, and grading judges the folder as the follow-up left it.
    if follow_up and status == "done":
        f_status, f_code, f_out, f_err = _agent_turn(job, oc, timeout, FOLLOW_UPS[follow_up])
        result.update({"follow_up": follow_up, "follow_up_status": f_status, "first_reply": result["reply"],
                       "follow_up_stdout_tail": f_out[-3000:], "follow_up_stderr_tail": f_err[-1500:]})
        result["reply"] = _reply_text(f_out) or result["reply"]
    if await_interventions:
        iv = _await_intervention(job["session_key"], log_offset, timeout)
        result["intervention"] = iv
        if iv.get("scheduled"):
            # The follow-up's reply supersedes the first turn's; grading looks
            # the latest one up from the session store.
            result["reply"] = None
            # A retried run that then finished counts as done: its work is on disk.
            if status != "done" and iv.get("action") == "retry" and iv.get("outcome_success"):
                result["status"] = "done"
                result["recovered_by"] = "retry"
    result["seconds"] = round(time.time() - started, 1)
    return result


def run(args) -> None:
    validate()
    batch_dir = BATCH_DIR / args.batch
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = batch_dir / "runs.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry["status"] == "done" or not args.retry_failed:
                done.add(entry["session_key"])
    jobs = [j for j in _jobs(args) if j["session_key"] not in done]
    meta_path = batch_dir / "batch.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "batch": args.batch, "created": datetime.now(timezone.utc).isoformat(),
            "models": args.models, "reps": args.reps, "only": args.only, "category": args.category,
            "follow_up": args.follow_up, "await_interventions": args.await_interventions,
        }, indent=1), encoding="utf-8")
    print(f"{len(jobs)} runs to go ({len(done)} already in the manifest)")
    if args.dry_run:
        for j in jobs[:20]:
            print(j["session_key"])
        return

    scenarios = {s["id"]: s for s in SCENARIOS}
    oc = _openclaw_cmd()
    lock = threading.Lock()
    counts: Counter = Counter()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_run_job, j, scenarios[j["scenario"]], oc, args.timeout, args.follow_up,
                               args.await_interventions): j for j in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with lock:
                with open(manifest, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
                counts[result["status"]] += 1
            elapsed = time.time() - t0
            eta = elapsed / i * (len(jobs) - i)
            print(f"[{i}/{len(jobs)}] {result['status']:<7} {result['scenario']:<22} {result['model']:<8} "
                  f"r{result['rep']} {result['seconds']:>6}s   eta {eta / 60:.0f}m   {counts}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--setup", action="store_true", help="register the suite's models in OpenClaw and exit")
    parser.add_argument("--batch", default="b" + datetime.now().strftime("%m%d%H%M"))
    parser.add_argument("--models", default="glm53f", help="comma list of short names, or 'all'")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--only", help="comma list of scenario ids")
    parser.add_argument("--category", help="comma list of categories")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600, help="per-run agent timeout, seconds")
    parser.add_argument("--retry-failed", action="store_true", help="re-run jobs whose last status wasn't done")
    parser.add_argument("--follow-up", choices=sorted(FOLLOW_UPS),
                        help="send this follow-up turn, in the same session, after every run that finishes")
    parser.add_argument("--await-interventions", action="store_true",
                        help="after each run, wait for the plugin's intervention decision and any follow-up turn it started")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.setup:
        setup_models()
    else:
        run(args)


if __name__ == "__main__":
    main()
