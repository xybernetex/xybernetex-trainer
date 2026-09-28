"""Can the supervisor tell, partway through a run, that it's going to fail -
and would stopping those runs early cut tokens per completed task without
losing completed tasks?

Trained and judged entirely offline on graded batches: every logged step's
snapshot (the v3 features the live policy sees, plus tokens spent so far)
labeled with whether its run ultimately failed. Evaluated by leaving whole
scenarios out, so the numbers describe tasks the predictor has never seen.
The stopping simulation then replays each held-out run: stop at the first
step (after --min-step) whose predicted failure risk crosses the threshold.
A stopped run counts as a failure and costs the tokens spent up to that
step; an unstopped run costs what it actually cost.

Usage:
    python -m scenarios.failure_predictor --batch-dir scenarios/batches/A --batch-dir scenarios/batches/B ...
"""
from __future__ import annotations

import argparse
import math
import random
import sqlite3
from bisect import bisect_right
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from chassis.action_space import ActionSpace
from chassis.state_vectorizer import vectorize_v3
from data.replay_loader import DEFAULT_LOG_PATH, load_runs
from scenarios.export_calls import _events
from scenarios.outcome_report import latest_jobs
from scenarios.outcomes import SESSION_DB
from scenarios.scorecard import load_outcomes

FAILED_RUN = {"failed", "timeout"}


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def token_timeline(conn, job) -> tuple[list[datetime], list[int]]:
    """Cumulative tokens after each assistant message of the latest attempt."""
    since = _ts(job["started"])
    stamps, totals, running = [], [], 0
    for e in _events(conn, job["session_key"]):
        msg = e.get("message") if e.get("type") == "message" else None
        if not msg or msg.get("role") != "assistant" or not isinstance(msg.get("usage"), dict) or not e.get("timestamp"):
            continue
        t = _ts(e["timestamp"])
        if t < since:
            continue
        running += msg["usage"].get("totalTokens", 0) or 0
        stamps.append(t)
        totals.append(running)
    return stamps, totals


MODELS = ["glm53f", "glm53", "glm47f", "dsv4f", "dsv4p", "qwen38", "nemo3", "gemma4", "kimi27c"]


def _model_onehot(model: str) -> np.ndarray:
    v = np.zeros(len(MODELS), dtype=np.float32)
    if model in MODELS:
        v[MODELS.index(model)] = 1.0
    return v


def _trajectory_features(steps, spent) -> list[np.ndarray]:
    """Whole-run history up to each step - the snapshot itself only carries
    the last 8 calls and 16 decisions."""
    out, failures, streak, max_streak, repeats, tools, prev = [], 0, 0, 0, 0, set(), None
    for k, (s, tokens) in enumerate(zip(steps, spent), 1):
        last = s.snapshot.tool_history[-1] if s.snapshot.tool_history else None
        if last is not None:
            failed = not last.success
            failures += failed
            streak = streak + 1 if failed else 0
            max_streak = max(max_streak, streak)
            signature = (last.tool_name, str(last.params))
            repeats += signature == prev
            prev = signature
            tools.add(last.tool_name)
        out.append(np.array([math.log1p(k) / 4, failures / k, min(max_streak, 10) / 10, min(streak, 10) / 10,
                             repeats / k, len(tools) / 10, math.log1p(tokens) / 12,
                             math.log1p(tokens / k) / 10], dtype=np.float32))
    return out


def collect(batch_dirs: list[Path], log_path: Path = DEFAULT_LOG_PATH) -> list[dict]:
    """One record per graded run: its steps (features) and token accounting."""
    runs = load_runs(log_path)
    by_session: dict[str, list] = {}
    for run in runs.values():
        by_session.setdefault(run.session_key, []).extend(run.steps)
    conn = sqlite3.connect(f"file:{SESSION_DB}?mode=ro", uri=True)
    records = []
    try:
        for batch_dir in batch_dirs:
            grades = load_outcomes(batch_dir)
            for job in latest_jobs(batch_dir):
                if job.get("status") in FAILED_RUN:
                    failed = True
                elif job.get("status") == "done" and grades.get(job["session_key"], {}).get("task") is not None:
                    failed = not grades[job["session_key"]]["task"]
                else:
                    continue
                since = _ts(job["started"])
                steps = sorted((s for s in by_session.get(job["session_key"], []) if _ts(s.ts) >= since
                                and all(a in ActionSpace.PRIMITIVES for a in s.snapshot.recent_actions)),
                               key=lambda s: s.snapshot.step_number)
                if not steps:
                    continue
                stamps, totals = token_timeline(conn, job)
                total_tokens = totals[-1] if totals else 0
                spent = [totals[bisect_right(stamps, _ts(s.ts)) - 1] if bisect_right(stamps, _ts(s.ts)) else 0
                         for s in steps]
                feats = [np.concatenate([vectorize_v3(s.snapshot), trajectory, _model_onehot(job["model"])])
                         for s, trajectory in zip(steps, _trajectory_features(steps, spent))]
                records.append({"batch": batch_dir.name, "scenario": job["scenario"], "model": job["model"],
                                "failed": failed, "features": np.stack(feats).astype(np.float32),
                                "spent": spent, "total_tokens": total_tokens})
    finally:
        conn.close()
    return records


class Predictor(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(records: list[dict], epochs: int = 60, seed: int = 0) -> Predictor:
    torch.manual_seed(seed)
    x = torch.tensor(np.concatenate([r["features"] for r in records]))
    y = torch.tensor(np.concatenate([[float(r["failed"])] * len(r["features"]) for r in records]), dtype=torch.float32)
    # Each run weighs the same however many steps it has; classes are balanced.
    w = np.concatenate([[1.0 / len(r["features"])] * len(r["features"]) for r in records])
    pos = sum(r["failed"] for r in records) or 1
    neg = (len(records) - pos) or 1
    w = w * np.concatenate([[len(records) / (2 * (pos if r["failed"] else neg))] * len(r["features"]) for r in records])
    w = torch.tensor(w / w.mean(), dtype=torch.float32)
    model = Predictor(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 256):
            idx = perm[i:i + 256]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(model(x[idx]), y[idx], reduction="none")
                    * w[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def cross_validate(records: list[dict], folds: int = 5, seed: int = 0) -> None:
    """Attach held-out per-step failure risk to every record, leaving whole
    scenarios out of each fold's training data."""
    scenarios = sorted({r["scenario"] for r in records})
    random.Random(seed).shuffle(scenarios)
    for k in range(folds):
        held = set(scenarios[k::folds])
        model = train([r for r in records if r["scenario"] not in held], seed=seed + k)
        with torch.no_grad():
            for r in records:
                if r["scenario"] in held:
                    r["risk"] = torch.sigmoid(model(torch.tensor(r["features"]))).numpy()


def auc(records: list[dict]) -> float:
    """Run-level AUC of the risk at each run's third step (or last, if shorter)."""
    scores = [(float(r["risk"][min(2, len(r["risk"]) - 1)]), r["failed"]) for r in records]
    pos = [s for s, f in scores if f]
    neg = [s for s, f in scores if not f]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def simulate(records: list[dict], threshold: float | None, min_step: int = 3) -> dict:
    tokens, wins, stopped, stopped_wins = 0, 0, 0, 0
    for r in records:
        stop_at = None
        if threshold is not None:
            for k, risk in enumerate(r["risk"]):
                if k + 1 >= min_step and risk >= threshold:
                    stop_at = k
                    break
        if stop_at is None:
            tokens += r["total_tokens"]
            wins += not r["failed"]
        else:
            tokens += r["spent"][stop_at]
            stopped += 1
            stopped_wins += not r["failed"]
    return {"tokens_per_success": tokens / wins if wins else float("inf"), "success": wins / len(records),
            "stopped": stopped, "stopped_would_succeed": stopped_wins, "tokens": tokens}


def best_safe_point(records: list[dict], max_success_loss: float, min_step: int = 3) -> dict | None:
    """The biggest cut in tokens per completed task among thresholds that lose
    at most max_success_loss of success. Cutting off successful runs also
    lowers tokens per completed task - by destroying value - so a saving only
    counts where success holds."""
    base = simulate(records, None)
    best = None
    for th in [x / 100 for x in range(30, 100)]:
        s = simulate(records, th, min_step)
        if base["success"] - s["success"] <= max_success_loss + 1e-9:
            saving = 1 - s["tokens_per_success"] / base["tokens_per_success"]
            if best is None or saving > best["saving"]:
                best = {**s, "threshold": th, "saving": saving, "success_loss": base["success"] - s["success"]}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch-dir", type=Path, action="append", required=True)
    parser.add_argument("--min-step", type=int, default=3)
    args = parser.parse_args()
    records = collect(args.batch_dir)
    print(f"{len(records)} runs ({sum(r['failed'] for r in records)} failed), "
          f"{sum(len(r['features']) for r in records)} steps, from {', '.join(d.name for d in args.batch_dir)}")
    cross_validate(records)
    print(f"held-out run-level AUC at step 3: {auc(records):.3f}\n")
    for label, subset in [("all", records)] + [(b.name, [r for r in records if r["batch"] == b.name]) for b in args.batch_dir]:
        base = simulate(subset, None)
        print(f"== {label}: {len(subset)} runs, no stopping: success {base['success']:.1%}, "
              f"tokens/completed task {base['tokens_per_success']:,.0f}")
        for loss in (0.0, 0.005, 0.01, 0.02):
            b = best_safe_point(subset, loss, args.min_step)
            if b is None or b["saving"] <= 0:
                print(f"   losing at most {loss:.1%} of success: no token saving available")
            else:
                print(f"   losing at most {loss:.1%} of success: best saving {b['saving']:.1%} of tokens per completed "
                      f"task at risk >= {b['threshold']:.2f} (success {b['success']:.1%}; stopped {b['stopped']} runs, "
                      f"{b['stopped_would_succeed']} would have succeeded)")


if __name__ == "__main__":
    main()
