"""One command for a finished intervention experiment: grade each arm once,
compare the arms pair by pair, break results down by model, check that the
interventions actually ran as designed, then fit, evaluate and export the
learned policy.

  arms     scenarios/batches/<prefix>-{none,always,targeted} over the same jobs
  graded   each arm's outcomes.jsonl is written once and reused (grading runs
           every checker in the sandbox, so it's the slow part)
  output   one Markdown report (and a .json twin), plus the learned table for
           /intervene when --export is given

    python -m scenarios.arm_results                                  # hard2 arms
    python -m scenarios.arm_results --prefix hard3-arm --export out/intervene_policy.json

Nothing here changes the batches, the gateway or the live policy: uploading
the exported table is a separate, deliberate step
(xybernetex-cfworker scripts/upload_intervene_policy.js).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from data.replay_loader import DEFAULT_LOG_PATH
from learn import bandit
from scenarios.outcome_report import grade_batch, latest_jobs
from scenarios.scorecard import compare, job_rows, summarize

ARMS = ("none", "always", "targeted")
PAIRS = (("none", "always"), ("none", "targeted"), ("always", "targeted"))


def ensure_graded(batch_dir: Path, regrade: bool = False) -> int:
    """Writes outcomes.jsonl unless it's already there; returns jobs graded."""
    path = batch_dir / "outcomes.jsonl"
    if path.exists() and not regrade:
        return 0
    rows = grade_batch(batch_dir)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def by_model(rows: list[dict]) -> dict[str, dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["key"][1]].append(r)
    return {m: summarize(rs) for m, rs in sorted(groups.items())}


def intervention_health(batch_dir: Path) -> dict:
    """What the plugin decided and did for each job, from the harness manifest."""
    decided, started, finished, problems = Counter(), Counter(), Counter(), Counter()
    by_group = defaultdict(Counter)  # model group -> action taken on finished runs
    rescued = 0
    for j in latest_jobs(batch_dir):
        iv = j.get("intervention") or {}
        if not iv.get("action"):
            problems[iv.get("note") or "no intervention record"] += 1
            continue
        decided[iv["action"]] += 1
        # The always arm uses the local rule by design; only a fallback is a problem.
        if str(iv.get("rule", "")).startswith("fallback:"):
            problems["policy service unreachable (local fallback)"] += 1
        if iv.get("scheduled"):
            started[iv["action"]] += 1
            if iv.get("outcome_success") is None:
                problems[iv.get("note") or "follow-up outcome missing"] += 1
            elif iv.get("outcome_success"):
                finished[iv["action"]] += 1
        if j.get("recovered_by") == "retry":
            rescued += 1
        if iv["action"] in ("verify", "none") and j.get("status") == "done":
            by_group[bandit.model_group(iv.get("model") or j["model"])][
                iv["action"] if iv.get("scheduled") or iv["action"] == "none" else "verify (not started)"] += 1
    return {"decided": dict(decided), "started": dict(started), "finished_cleanly": dict(finished),
            "rescued_by_retry": rescued, "problems": dict(problems),
            "verify_by_group": {g: dict(c) for g, c in sorted(by_group.items())}}


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def _num(x, fmt="{:,.0f}"):
    return "n/a" if x is None else fmt.format(x)


def _ci(v):
    return "" if not v else f" ({v[0]:.2f}-{v[1]:.2f})"


def render(prefix: str, arms: list[str], summaries: dict, comparisons: dict, models: dict, health: dict,
           bandit_text: str | None, bandit_note: str | None) -> str:
    out = [f"# {prefix}: intervention arms", ""]
    out += ["## The four numbers, per arm", "", "| | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for label, fn in (("Task success", lambda s: f"{_pct(s['success_rate'])} ({s['successes']}/{s['scored']})"),
                      ("Tokens per completed task", lambda s: _num(s["tokens_per_success"])),
                      ("Seconds per completed task", lambda s: _num(s["seconds_per_success"], "{:,.1f}")),
                      ("False blocks on successful runs", lambda s: f"{s['false_block_runs']}/{s['clean_successes']}")):
        out.append(f"| {label} | " + " | ".join(fn(summaries[a]) for a in arms) + " |")
    out += ["", "## Paired comparisons (same scenario, model and repeat)", "",
            "| | pairs | success | only B completed | only A completed | McNemar p | tokens/completed x (95% CI) | seconds/completed x (95% CI) |",
            "|---|---|---|---|---|---|---|---|"]
    for (a, b), c in comparisons.items():
        out.append(f"| {b} vs {a} | {c['pairs']} | {_pct(c['a_success'])} -> {_pct(c['b_success'])} | {c['only_b']} | "
                   f"{c['only_a']} | {c['mcnemar_p']:.3f} | {_num(c['tokens_ratio'], '{:.2f}')}{_ci(c['tokens_ci'])} | "
                   f"{_num(c['seconds_ratio'], '{:.2f}')}{_ci(c['seconds_ci'])} |")
    out += ["", "## By model", "", "| model | " + " | ".join(f"{a} success | {a} tok/completed" for a in arms) + " |",
            "|---|" + "---|---|" * len(arms)]
    for m in sorted({m for a in arms for m in models[a]}):
        cells = []
        for a in arms:
            s = models[a].get(m)
            cells += ([f"{_pct(s['success_rate'])} ({s['successes']}/{s['scored']})", _num(s["tokens_per_success"])]
                      if s else ["-", "-"])
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out += ["", "## Did the interventions run as designed?", ""]
    for a in arms:
        h = health[a]
        out.append(f"- **{a}**: decided {h['decided']}; started {h['started']}; follow-ups finished cleanly "
                   f"{h['finished_cleanly']}; dead runs rescued by retry {h['rescued_by_retry']}"
                   + (f"; problems {h['problems']}" if h["problems"] else "; no problems"))
    if "targeted" in health:
        out += ["", "Targeted arm, finished runs by model group (from /intervene):", ""]
        for g, c in health["targeted"]["verify_by_group"].items():
            n = sum(c.values())
            out.append(f"- {g}: verified {c.get('verify', 0)}/{n} ({_pct(c.get('verify', 0) / n if n else None)})")
    out += ["", "## Learned policy (off-policy estimates, cross-fitted by scenario)", ""]
    out += [bandit_note] if bandit_note else []
    out += ["```", bandit_text, "```"] if bandit_text else []
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("."), help="trainer repo holding scenarios/batches")
    parser.add_argument("--prefix", default="hard2-arm")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--regrade", action="store_true", help="grade again even if outcomes.jsonl exists")
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--export", type=Path, help="write the learned /intervene table here")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dirs = {a: args.root / "scenarios" / "batches" / f"{args.prefix}-{a}" for a in ARMS}
    arms = [a for a in ARMS if (dirs[a] / "runs.jsonl").exists()]
    if len(arms) < 2:
        raise SystemExit(f"need at least two arms under {args.root / 'scenarios' / 'batches'}; found {arms}")
    for a in arms:
        n = ensure_graded(dirs[a], args.regrade)
        print(f"{a}: {'graded ' + str(n) + ' jobs' if n else 'using stored grades'}", flush=True)

    rows = {a: job_rows(dirs[a], args.log) for a in arms}
    summaries = {a: summarize(rows[a]) for a in arms}
    comparisons = {(a, b): compare(rows[a], rows[b]) for a, b in PAIRS if a in rows and b in rows}
    models = {a: by_model(rows[a]) for a in arms}
    health = {a: intervention_health(dirs[a]) for a in arms}

    bandit_text, bandit_note, results, table = None, None, None, None
    records = bandit.load(args.root, args.log, batch=f"scenarios/batches/{args.prefix}-{{}}")
    sup = bandit.support(records)
    thin = [k for k, v in sup.items() if v < 5]
    bandit_note = "Logged examples: " + ", ".join(f"{k} {v}" for k, v in sup.items()) + "."
    if len(records) < 20 or "finished/verify" in thin:
        bandit_note += f" Too few to evaluate ({len(records)} usable runs)."
    else:
        if thin:
            bandit_note += f" Thin support for {', '.join(thin)}: estimates there lean on the reward model."
        bandit_text, results = bandit.report(records, args.lam, 0.0, args.rounds)
        if args.export:
            table = bandit.export_policy(records, args.lam, args.epsilon)
            args.export.parent.mkdir(parents=True, exist_ok=True)
            args.export.write_text(json.dumps(table, indent=1), encoding="utf-8")
            bandit_note += f" Learned table written to {args.export} (not uploaded)."

    markdown = render(args.prefix, arms, summaries, comparisons, models, health, bandit_text, bandit_note)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        args.output.with_suffix(".json").write_text(json.dumps({
            "summaries": summaries, "comparisons": {f"{b}_vs_{a}": c for (a, b), c in comparisons.items()},
            "by_model": models, "interventions": health, "bandit": results, "learned_table": table,
        }, indent=2, default=str), encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
