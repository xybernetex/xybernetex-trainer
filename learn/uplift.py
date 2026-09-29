"""Learn which follow-up, if any, each run should get - from counterfactual
batches (scenarios/run_suite.py --counterfactual). Every run carries the
outcome of EVERY action from the same starting point: no follow-up (graded as
the first turn left it) and each follow-up branch (verify in the same or a
fresh session, a test-writing check, verify on the strong model; for dead
runs, retries). So a policy is scored exactly on each run: no propensities,
no noise from comparing different runs.

Under "none" a run keeps its first-turn grade and tokens; under an action it
gets that branch's grade and pays that branch's tokens too. The objective is
utility = success - lam * tokens / 100k per run (lam 0.05: a 5-point success
gain is worth 100k tokens).

Candidate policies, all fit only on training data:
  never, always:<action>  the fixed rules, one per action
  v0-targeted             the live /intervene rule (same-session verify by model group)
  per-model               each model gets the action with the best mean utility
  failure-threshold       logistic P(first answer wrong | features); above a
                          threshold (inner cross-validation) take the best
                          action on average
  uplift-logistic         per action, P(fixed) - P(broken) minus its expected
                          token cost; take the best if positive
  uplift-tree             per action, a depth-2 regression tree on the utility
                          gain; take the best if positive

Scoring: leave-one-task-out (a policy never sees the task it's scored on),
with a bootstrap over tasks against the best fixed rule; then, with --test,
train on all of --train and score once on the held-out tasks.

    python -m learn.uplift --train scenarios/batches/cf-hard2 --test scenarios/batches/cf-hard3
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from learn.bandit import model_group
from scenarios.cf_features import FEATURE_NAMES

V0_ACTION = {"finished": "verify-same", "died": "retry-same"}
VERIFY_HELPS_V0 = {"helps": True, "wastes": False, "unknown": True}  # v0-targeted at its most likely draw


@dataclass
class Run:
    scenario: str
    model: str
    kind: str                 # "finished" | "died"
    y0: float                 # success with no follow-up
    t0: float                 # tokens of the first turn
    x: np.ndarray             # decision-time features
    arms: dict = field(default_factory=dict)  # action -> (success, follow-up tokens)

    def outcome(self, action: str) -> tuple[float, float]:
        return (self.y0, self.t0) if action == "none" else (self.arms[action][0], self.t0 + self.arms[action][1])


def _model_id(short: str) -> str:
    """The harness records short names (glm53f); /intervene and model_group see provider/model ids."""
    from scenarios.run_suite import MODELS, PROVIDER
    return f"{PROVIDER}/{MODELS[short][0]}" if short in MODELS else short


def load(batch_dirs: list[Path]) -> list[Run]:
    """Runs whose first turn and every branch have a grade; others are left out."""
    runs = []
    for d in batch_dirs:
        latest = {}
        for line in (d / "runs.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                j = json.loads(line)
                latest[j.get("base_key", j["session_key"])] = j
        for j in latest.values():
            cf = j.get("counterfactual") or {}
            branches, feats = cf.get("branches") or {}, cf.get("features")
            if cf.get("kind") not in ("finished", "died") or feats is None or not branches:
                continue
            if (cf.get("before") or {}).get("task") is None or any(b.get("task") is None for b in branches.values()):
                continue
            runs.append(Run(scenario=j["scenario"], model=_model_id(j["model"]), kind=cf["kind"],
                            y0=float(cf["before"]["task"]), t0=float(cf.get("tokens_first") or 0),
                            x=np.array([float(feats.get(k, 0.0)) for k in FEATURE_NAMES]),
                            arms={a: (float(b["task"]), float(b.get("tokens") or 0)) for a, b in branches.items()}))
    return runs


# --- scoring -------------------------------------------------------------------------------

def score(runs: list[Run], actions: list[str], lam: float) -> dict:
    ys, ts = zip(*(r.outcome(a) for r, a in zip(runs, actions))) if runs else ((), ())
    y, t = np.array(ys), np.array(ts)
    return {"n": len(runs), "success": float(y.mean()) if len(y) else float("nan"),
            "tokens_per_success": float(t.sum() / y.sum()) if y.sum() else float("inf"),
            "utility": float((y - lam * t / 100_000).mean()) if len(y) else float("nan"),
            "followed_up": float(np.mean([a != "none" for a in actions])) if actions else 0.0}


def gain(r: Run, action: str, lam: float) -> float:
    """Utility of taking `action` on this run minus taking none."""
    y, t = r.arms[action]
    return (y - r.y0) - lam * t / 100_000


# --- learners (numpy only) --------------------------------------------------------------------

class Standardizer:
    def __init__(self, X: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd


def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1.0, iters: int = 50) -> np.ndarray:
    """L2-penalized logistic regression by Newton's method; returns weights with a bias first."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    pen = np.eye(Xb.shape[1]) * ridge
    pen[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + pen @ w
        H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + pen
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def predict_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(np.hstack([np.ones((len(X), 1)), X]) @ w)))


def fit_tree(X: np.ndarray, target: np.ndarray, depth: int = 2, min_leaf: int = 8):
    """Regression tree on a per-run gain; a leaf predicts its mean."""
    if depth == 0 or len(target) < 2 * min_leaf:
        return float(target.mean()) if len(target) else 0.0
    best = None
    for f in range(X.shape[1]):
        for thr in np.unique(np.quantile(X[:, f], [0.2, 0.35, 0.5, 0.65, 0.8])):
            left = X[:, f] <= thr
            nl, nr = left.sum(), (~left).sum()
            if nl < min_leaf or nr < min_leaf:
                continue
            sse = ((target[left] - target[left].mean()) ** 2).sum() + ((target[~left] - target[~left].mean()) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, f, thr, left)
    if best is None:
        return float(target.mean())
    _, f, thr, left = best
    return (f, thr, fit_tree(X[left], target[left], depth - 1, min_leaf), fit_tree(X[~left], target[~left], depth - 1, min_leaf))


def predict_tree(tree, x: np.ndarray) -> float:
    while isinstance(tree, tuple):
        f, thr, lo, hi = tree
        tree = lo if x[f] <= thr else hi
    return tree


def _design(runs: list[Run], models: list[str]) -> np.ndarray:
    onehot = np.array([[float(r.model == m) for m in models] for r in runs]) if models else np.zeros((len(runs), 0))
    return np.hstack([np.array([r.x for r in runs]).reshape(len(runs), -1), onehot])


def actions_of(runs: list[Run]) -> list[str]:
    return sorted({a for r in runs for a in r.arms})


def fixed_policies(runs: list[Run]) -> dict:
    pols = {"never": lambda r: "none"}
    for a in actions_of(runs):
        pols[f"always:{a}"] = (lambda a: lambda r: a)(a)
    pols["v0-targeted"] = lambda r: (V0_ACTION[r.kind] if r.kind == "died" or VERIFY_HELPS_V0[model_group(r.model)]
                                     else "none")
    return pols


def train_policies(train: list[Run], lam: float) -> dict:
    """Every candidate, fit on `train` (one kind of run), as run -> action."""
    pols = fixed_policies(train)
    acts = actions_of(train)
    models = sorted({r.model for r in train})
    std = Standardizer(_design(train, models))
    Z = std(_design(train, models))

    def z_of(r: Run) -> np.ndarray:
        return std(_design([r], models))[0]

    mean_gain = {a: float(np.mean([gain(r, a, lam) for r in train])) for a in acts}
    best_avg = max(acts, key=lambda a: mean_gain[a])

    by_model = defaultdict(list)
    for r in train:
        by_model[r.model].append(r)
    per_model = {}
    for m, rs in by_model.items():
        u = {a: float(np.mean([gain(r, a, lam) for r in rs])) for a in acts}
        a = max(u, key=u.get)
        per_model[m] = a if u[a] > 0 else "none"
    pols["per-model"] = lambda r: per_model.get(r.model, best_avg if mean_gain[best_avg] > 0 else "none")

    trees = {a: fit_tree(Z, np.array([gain(r, a, lam) for r in train])) for a in acts}
    t_cost = {a: defaultdict(list) for a in acts}
    for r in train:
        for a in acts:
            t_cost[a][r.model].append(r.arms[a][1])
    t_mean = {a: {m: float(np.mean(v)) for m, v in c.items()} for a, c in t_cost.items()}
    t_all = {a: float(np.mean([r.arms[a][1] for r in train])) for a in acts}
    w_fix = {a: fit_logistic(Z, np.array([float(r.arms[a][0] > r.y0) for r in train])) for a in acts}
    w_brk = {a: fit_logistic(Z, np.array([float(r.arms[a][0] < r.y0) for r in train])) for a in acts}

    def argmax_positive(scores: dict) -> str:
        a = max(scores, key=scores.get)
        return a if scores[a] > 0 else "none"

    pols["uplift-tree"] = lambda r: argmax_positive({a: predict_tree(trees[a], z_of(r)) for a in acts})
    pols["uplift-logistic"] = lambda r: argmax_positive({
        a: predict_logistic(w_fix[a], z_of(r)[None])[0] - predict_logistic(w_brk[a], z_of(r)[None])[0]
        - lam * t_mean[a].get(r.model, t_all[a]) / 100_000 for a in acts})

    # Failure threshold: t by inner leave-one-task-out, then the best action on average.
    wrong = np.array([float(r.y0 == 0) for r in train])
    w_wrong = fit_logistic(Z, wrong)
    grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    inner = {t: 0.0 for t in grid}
    for s in sorted({r.scenario for r in train}):
        tr = [r for r in train if r.scenario != s]
        te = [r for r in train if r.scenario == s]
        if len({r.scenario for r in tr}) < 2 or not te:
            continue
        st = Standardizer(_design(tr, models))
        w = fit_logistic(st(_design(tr, models)), np.array([float(r.y0 == 0) for r in tr]))
        p = predict_logistic(w, st(_design(te, models)))
        for t in grid:
            inner[t] += sum(gain(r, best_avg, lam) for r, pi in zip(te, p) if pi > t)
    t_best = max(grid, key=lambda t: inner[t])
    pols["failure-threshold"] = lambda r: best_avg if predict_logistic(w_wrong, z_of(r)[None])[0] > t_best else "none"
    return pols


def fixed_names(runs: list[Run]) -> list[str]:
    return list(fixed_policies(runs))


def crossfit(runs: list[Run], lam: float) -> dict[str, list[str]]:
    """Leave-one-task-out: each run's action comes from policies fit without its task."""
    out = defaultdict(lambda: ["none"] * len(runs))
    for s in sorted({r.scenario for r in runs}):
        train = [r for r in runs if r.scenario != s]
        pols = train_policies(train, lam)
        for i, r in enumerate(runs):
            if r.scenario == s:
                for name, pi in pols.items():
                    out[name][i] = pi(r)
    return dict(out)


def bootstrap_gain(runs: list[Run], decisions: dict[str, list[str]], lam: float, rounds: int = 1000, seed: int = 0) -> dict:
    """95% interval, over tasks, of each policy's utility minus the best fixed rule's (chosen per resample)."""
    rng = random.Random(seed)
    fixed = set(fixed_names(runs))
    by = defaultdict(list)
    for i, r in enumerate(runs):
        by[r.scenario].append(i)
    tasks = sorted(by)
    diffs = defaultdict(list)
    for _ in range(rounds):
        idx = [i for s in (rng.choice(tasks) for _ in tasks) for i in by[s]]
        sub = [runs[i] for i in idx]
        util = {k: score(sub, [v[i] for i in idx], lam)["utility"] for k, v in decisions.items()}
        best_fixed = max(u for k, u in util.items() if k in fixed)
        for k, u in util.items():
            diffs[k].append(u - best_fixed)
    return {k: (sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]) for k, v in diffs.items()}


def _section(runs: list[Run], test: list[Run] | None, lam: float, rounds: int, label: str) -> list[str]:
    lines = [f"## {label}: {len(runs)} runs over {len({r.scenario for r in runs})} tasks", ""]
    lines.append(f"{'action':16s} {'success':>8s} {'fixed':>6s} {'broke':>6s} {'follow-up tokens':>17s}")
    lines.append(f"{'none':16s} {np.mean([r.y0 for r in runs]):8.1%}")
    for a in actions_of(runs):
        lines.append(f"{a:16s} {np.mean([r.arms[a][0] for r in runs]):8.1%} {sum(r.arms[a][0] > r.y0 for r in runs):6d} "
                     f"{sum(r.arms[a][0] < r.y0 for r in runs):6d} {np.mean([r.arms[a][1] for r in runs]):17,.0f}")
    if len({r.scenario for r in runs}) < 3:
        return lines + ["", "(too few tasks to cross-fit)"]
    dec = crossfit(runs, lam)
    ci = bootstrap_gain(runs, dec, lam, rounds)
    fixed = fixed_names(runs)
    best_fixed = max(score(runs, dec[k], lam)["utility"] for k in fixed)
    lines += ["", "Leave-one-task-out:", "",
              f"{'policy':22s} {'success':>8s} {'tok/completed':>14s} {'utility':>8s} {'follow-up':>9s}  vs best fixed (95% CI)"]
    for k, v in dec.items():
        s = score(runs, v, lam)
        lines.append(f"{k:22s} {s['success']:8.1%} {s['tokens_per_success']:14,.0f} {s['utility']:8.3f} {s['followed_up']:9.0%}  "
                     f"{s['utility'] - best_fixed:+.3f} ({ci[k][0]:+.3f} to {ci[k][1]:+.3f})")
    if test:
        pols = train_policies(runs, lam)
        lines += ["", f"Held-out tasks: {len(test)} runs over {len({r.scenario for r in test})} tasks, trained on all the above:", ""]
        for name, pi in pols.items():
            s = score(test, [pi(r) for r in test], lam)
            lines.append(f"{name:22s} {s['success']:8.1%} {s['tokens_per_success']:14,.0f} {s['utility']:8.3f} {s['followed_up']:9.0%}")
    return lines


def report(train: list[Run], test: list[Run] | None, lam: float, rounds: int) -> str:
    lines = [f"lam={lam}: utility = success - {lam} x tokens/100k per run", ""]
    for kind, label in (("finished", "Finished runs"), ("died", "Dead runs")):
        tr = [r for r in train if r.kind == kind]
        te = [r for r in test if r.kind == kind] if test else None
        if tr:
            lines += _section(tr, te, lam, rounds, label) + [""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="*")
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = report(load(args.train), load(args.test) if args.test else None, args.lam, args.rounds)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
