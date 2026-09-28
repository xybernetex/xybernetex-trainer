"""Contextual bandit for follow-up interventions: learn, per run, whether a
finished run should get a "check your work" turn and a dead run a retry, and
estimate - from logged runs only - how any such policy would score.

Data: the hard2 intervention arms (scenarios/batches/hard2-arm-{none,always,
targeted}), each a third of the runs. Behaviour propensities come from
knowing each arm's rule exactly, pooled as a mixture (every run could have
come from any arm): pi_b(a|x) = mean over arms of pi_arm(a|x). That makes
every run usable for evaluating every candidate policy.

Context (what /intervene sees at decision time): did the first turn finish,
was a failure retriable, tool-call count, model. Actions: finished runs
{none, verify}; dead runs {none, retry}. Outcomes are graded task success,
tokens (all turns) and seconds.

Estimators: IPS, self-normalised IPS, direct method and doubly robust, with
the reward model cross-fitted by scenario so no run scores itself. The
learned policy is chosen the same cross-fitted way, so its estimate is
honest. It has to beat the best fixed rule, not just "never".

    python -m learn.bandit                       # evaluate on the three arms
    python -m learn.bandit --lam 0.05 --export out/intervene_policy.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from data.replay_loader import DEFAULT_LOG_PATH
from scenarios.outcome_report import latest_jobs
from scenarios.scorecard import job_rows

ARMS = ("none", "always", "targeted")
BATCH = "scenarios/batches/hard2-arm-{}"

# Mirror of xybernetex-cfworker/src/intervene.js (v0-targeted-2026-09-28).
VERIFY_HELPS = [re.compile(p, re.I) for p in (r"gemma-4", r"glm-4\.7", r"kimi-k2\.7", r"qwen3\.8", r"deepseek-v4-(pro|flash)")]
VERIFY_WASTES = [re.compile(p, re.I) for p in (r"nemotron", r"glm-5\.3")]
P_RETRY = 0.9
P_VERIFY = {"helps": 0.9, "wastes": 0.1, "unknown": 0.5}
RETRIABLE_RULES = {"retry-on-death", "retry-held-out", "v0-retry-on-death", "fallback:v0-retry-on-death"}


def model_group(model: str | None) -> str:
    if not model:
        return "unknown"
    if any(r.search(model) for r in VERIFY_HELPS):
        return "helps"
    if any(r.search(model) for r in VERIFY_WASTES):
        return "wastes"
    return "unknown"


@dataclass
class Record:
    arm: str
    scenario: str
    model: str
    finished: bool        # did the first turn end cleanly
    retriable: bool       # for dead runs: a failure worth retrying
    tool_calls: int
    action: str           # what actually happened: none | verify | retry
    success: bool         # graded task success after any follow-up
    tokens: int
    seconds: float
    behavior: dict = field(default_factory=dict)  # pooled pi_b(a|x)

    @property
    def actions(self) -> tuple[str, ...]:
        return ("none", "verify") if self.finished else ("none", "retry")


# --- the three arms' rules, so propensities are known exactly ---------------------

def arm_probs(arm: str, r: Record) -> dict[str, float]:
    """pi_arm(a | x) for each available action."""
    none_only = {a: float(a == "none") for a in r.actions}
    if arm == "none":
        return none_only
    if not r.finished:
        if not r.retriable:
            return none_only
        p = 1.0 if arm == "always" else P_RETRY
        return {"none": 1 - p, "retry": p}
    if r.tool_calls < 1:
        return none_only
    p = 1.0 if arm == "always" else P_VERIFY[model_group(r.model)]
    return {"none": 1 - p, "verify": p}


def pooled(r: Record, weights: dict[str, float] | None = None) -> dict[str, float]:
    """pi_b(a|x) for the pooled data: each arm's rule, weighted by that arm's
    share of the runs (equal shares when the experiment is complete). Valid
    because every arm ran the same job grid, so arm is independent of context."""
    weights = weights or {arm: 1 / len(ARMS) for arm in ARMS}
    return {a: sum(w * arm_probs(arm, r)[a] for arm, w in weights.items()) for a in r.actions}


# --- loading ----------------------------------------------------------------------

def _decisions(log_path: Path) -> dict[str, dict]:
    """Per session: the plugin's (non-outcome) intervention decision line."""
    out = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if '"type":"intervention"' not in line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("action") != "outcome":
                out[e["sessionKey"]] = e
    return out


def load(root: Path = Path("."), log_path: Path = DEFAULT_LOG_PATH) -> list[Record]:
    decisions = _decisions(log_path)
    records = []
    for arm in ARMS:
        batch = root / BATCH.format(arm)
        if not (batch / "runs.jsonl").exists():
            continue
        rows = {r["key"]: r for r in job_rows(batch, log_path)}
        for j in latest_jobs(batch):
            row = rows.get((j["scenario"], j["model"], j["rep"]))
            d = decisions.get(j["session_key"])
            iv = j.get("intervention") or {}
            if row is None or row["success"] is None or d is None:
                continue  # ungraded, harness crash, or no decision logged
            action = iv.get("action") if iv.get("scheduled") else "none"
            rec = Record(arm=arm, scenario=j["scenario"], model=d.get("model") or j["model"],
                         finished=bool(d.get("success")), retriable=d.get("rule") in RETRIABLE_RULES,
                         tool_calls=int(d.get("toolCalls") or 0), action=action or "none",
                         success=bool(row["success"]), tokens=int(row["tokens"]), seconds=float(row["seconds"]))
            records.append(rec)
    counts = {arm: sum(r.arm == arm for r in records) for arm in ARMS}
    weights = {arm: n / len(records) for arm, n in counts.items() if n} if records else {}
    kept = []
    for rec in records:
        rec.behavior = pooled(rec, weights)
        if rec.behavior.get(rec.action, 0) > 0:
            kept.append(rec)  # else: an action no arm could have taken here (e.g. a scheduling glitch)
    return kept


def support(records: list[Record]) -> dict[str, int]:
    """Logged examples per (first turn, action): an action with none can't be
    evaluated or learned, only guessed by the reward model's back-off."""
    out = {}
    for fin, acts in ((True, ("none", "verify")), (False, ("none", "retry"))):
        for a in acts:
            out[f"{'finished' if fin else 'died'}/{a}"] = sum(r.finished == fin and r.action == a for r in records)
    return out


# --- reward model and estimators ---------------------------------------------------

def _cells(r: Record, a: str) -> list[tuple]:
    """Back-off keys, most specific first."""
    return [(r.finished, r.model, a), (r.finished, model_group(r.model), a), (r.finished, a)]


class RewardModel:
    """Shrunk cell means of success and tokens per (finished, model, action),
    backing off to the model group and then to (finished, action)."""

    def __init__(self, records: list[Record], shrink: float = 3.0):
        self.shrink = shrink
        self.sums = defaultdict(lambda: [0, 0.0, 0.0])  # n, success, tokens
        for r in records:
            for key in _cells(r, r.action):
                s = self.sums[key]
                s[0] += 1
                s[1] += r.success
                s[2] += r.tokens

    def predict(self, r: Record, a: str) -> tuple[float, float]:
        keys = _cells(r, a)
        est = None
        for key in reversed(keys):  # coarse -> fine, each shrunk toward the coarser one
            n, succ, tok = self.sums.get(key, (0, 0.0, 0.0))
            if est is None:
                est = (succ / n, tok / n) if n else (0.5, 0.0)
            else:
                w = n / (n + self.shrink)
                est = (w * (succ / n if n else 0) + (1 - w) * est[0], w * (tok / n if n else 0) + (1 - w) * est[1])
        return est


def _folds(records: list[Record], k: int, seed: int) -> list[set[str]]:
    scenarios = sorted({r.scenario for r in records})
    random.Random(seed).shuffle(scenarios)
    return [set(scenarios[i::k]) for i in range(k)]


def evaluate(records: list[Record], policy, models: list[RewardModel] | None = None, folds=None) -> dict:
    """Estimates of success rate and tokens per run for `policy(r) -> {a: p}`.
    models/folds: cross-fitted reward models (model i was fit without fold i)."""
    out = {"ips_s": 0.0, "ips_t": 0.0, "w": 0.0, "wr_s": 0.0, "wr_t": 0.0, "dm_s": 0.0, "dm_t": 0.0, "dr_s": 0.0, "dr_t": 0.0}
    for r in records:
        pi = policy(r)
        w = pi.get(r.action, 0.0) / r.behavior[r.action]
        out["ips_s"] += w * r.success
        out["ips_t"] += w * r.tokens
        out["w"] += w
        out["wr_s"] += w * r.success
        out["wr_t"] += w * r.tokens
        if models is not None:
            m = models[next(i for i, f in enumerate(folds) if r.scenario in f)]
            q = {a: m.predict(r, a) for a in r.actions}
            dm_s = sum(pi.get(a, 0) * q[a][0] for a in r.actions)
            dm_t = sum(pi.get(a, 0) * q[a][1] for a in r.actions)
            out["dm_s"] += dm_s
            out["dm_t"] += dm_t
            out["dr_s"] += dm_s + w * (r.success - q[r.action][0])
            out["dr_t"] += dm_t + w * (r.tokens - q[r.action][1])
    n = len(records)
    res = {"n": n, "ips_success": out["ips_s"] / n, "snips_success": out["wr_s"] / out["w"] if out["w"] else float("nan")}
    res["snips_tokens"] = out["wr_t"] / out["w"] if out["w"] else float("nan")
    if models is not None:
        res.update({"dm_success": out["dm_s"] / n, "dr_success": out["dr_s"] / n, "dr_tokens": out["dr_t"] / n})
        res["dr_tokens_per_success"] = res["dr_tokens"] / res["dr_success"] if res["dr_success"] > 0 else float("inf")
    return res


# --- candidate policies -------------------------------------------------------------

def never(r: Record) -> dict:
    return {a: float(a == "none") for a in r.actions}


def always(r: Record) -> dict:
    return arm_probs("always", r)


def targeted_v0(r: Record) -> dict:
    return arm_probs("targeted", r)


def learned_policy(model: RewardModel, lam: float, epsilon: float = 0.0):
    """Greedy on predicted success minus lam per 100k tokens; epsilon keeps
    exploring (the deployed version must, so learning continues)."""
    def pi(r: Record) -> dict:
        value = {a: model.predict(r, a)[0] - lam * model.predict(r, a)[1] / 100_000 for a in r.actions}
        best = max(r.actions, key=lambda a: (value[a], a == "none"))  # ties go to doing nothing
        return {a: (1 - epsilon) * (a == best) + epsilon / len(r.actions) for a in r.actions}
    return pi


def crossfit(records: list[Record], k: int = 5, seed: int = 0):
    folds = _folds(records, k, seed)
    models = [RewardModel([r for r in records if r.scenario not in f]) for f in folds]
    return folds, models


def learned_crossfit(records: list[Record], folds, models, lam: float, epsilon: float = 0.0):
    """The learned policy, with each run's action chosen by a model that never
    saw that run's scenario - so evaluating it is out-of-sample."""
    per_fold = [learned_policy(m, lam, epsilon) for m in models]

    def pi(r: Record) -> dict:
        return per_fold[next(i for i, f in enumerate(folds) if r.scenario in f)](r)
    return pi


def bootstrap(records: list[Record], policy_factory, rounds: int = 300, seed: int = 0) -> dict:
    """95% intervals for DR success and tokens per success, resampling scenarios."""
    rng = random.Random(seed)
    by_scn = defaultdict(list)
    for r in records:
        by_scn[r.scenario].append(r)
    scns = sorted(by_scn)
    s_vals, t_vals = [], []
    for _ in range(rounds):
        sample = [r for s in (rng.choice(scns) for _ in scns) for r in by_scn[s]]
        if len({r.scenario for r in sample}) < 3:
            continue
        folds, models = crossfit(sample, k=min(5, len({r.scenario for r in sample})), seed=rng.random())
        est = evaluate(sample, policy_factory(sample, folds, models), models, folds)
        s_vals.append(est["dr_success"])
        t_vals.append(est["dr_tokens_per_success"])
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))] if v else float("nan")  # noqa: E731
    return {"success": (q(s_vals, 0.025), q(s_vals, 0.975)), "tokens_per_success": (q(t_vals, 0.025), q(t_vals, 0.975))}


def report(records: list[Record], lam: float, epsilon: float, rounds: int) -> tuple[str, dict]:
    folds, models = crossfit(records)
    candidates = {
        "never": lambda recs, f, m: never,
        "always (retry + verify)": lambda recs, f, m: always,
        "v0-targeted (live rule)": lambda recs, f, m: targeted_v0,
        f"learned (lam={lam})": lambda recs, f, m: learned_crossfit(recs, f, m, lam, epsilon),
    }
    lines = [f"{len(records)} graded runs with a logged decision; by arm: "
             + ", ".join(f"{a} {sum(r.arm == a for r in records)}" for a in ARMS), "",
             f"{'policy':28s} {'success (DR)':>13s} {'95% CI':>15s} {'tok/completed':>14s} {'95% CI':>19s} {'SNIPS succ':>11s}"]
    results = {}
    for name, factory in candidates.items():
        est = evaluate(records, factory(records, folds, models), models, folds)
        ci = bootstrap(records, factory, rounds=rounds)
        results[name] = {**est, "ci": ci}
        lines.append(f"{name:28s} {est['dr_success']:13.1%} {ci['success'][0]:7.1%}-{ci['success'][1]:6.1%} "
                     f"{est['dr_tokens_per_success']:14,.0f} {ci['tokens_per_success'][0]:9,.0f}-{ci['tokens_per_success'][1]:,.0f} "
                     f"{est['snips_success']:11.1%}")
    return "\n".join(lines), results


def export_policy(records: list[Record], lam: float, epsilon: float) -> dict:
    """The learned table for /intervene: per model seen, P(verify) for finished
    runs and P(retry) for retriable dead runs (epsilon-greedy)."""
    model = RewardModel(records)
    pi = learned_policy(model, lam, epsilon)
    table = {}
    for m in sorted({r.model for r in records}):
        fin = Record("x", "x", m, True, False, 5, "none", False, 0, 0.0)
        dead = Record("x", "x", m, False, True, 5, "none", False, 0, 0.0)
        table[m] = {"verify": round(pi(fin)["verify"], 3), "retry": round(pi(dead)["retry"], 3)}
    return {"lam": lam, "epsilon": epsilon, "runs": len(records), "models": table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lam", type=float, default=0.05,
                        help="success points given up per 100k tokens saved (0.05 = 5 points)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="exploration kept in the exported policy")
    parser.add_argument("--rounds", type=int, default=300, help="bootstrap rounds")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    records = load()
    sup = support(records)
    print("logged examples: " + ", ".join(f"{k} {v}" for k, v in sup.items()))
    missing = [k for k, v in sup.items() if v < 5]
    if len(records) < 20 or "finished/verify" in missing:
        raise SystemExit(f"{len(records)} usable runs; too few to evaluate yet (thin: {', '.join(missing) or 'none'})")
    if missing:
        print(f"warning: thin support for {', '.join(missing)}; estimates there lean on the reward model")
    text, _ = report(records, args.lam, 0.0, args.rounds)
    print(text)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(export_policy(records, args.lam, args.epsilon), indent=1), encoding="utf-8")
        print(f"\nwrote {args.export}")


if __name__ == "__main__":
    main()
