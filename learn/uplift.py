"""Learn when a follow-up turn pays off, from counterfactual batches
(scenarios/run_suite.py --counterfactual): every run carries BOTH outcomes -
graded as the first turn left it, and again after a harness follow-up - so a
policy is scored exactly on each run, with no propensities and no noise from
comparing different runs.

  finished runs  actions none | verify   (the learning problem)
  dead runs      actions none | retry    (reported: rescue rate and cost)

Under "none" a run keeps its first-turn grade and tokens; under the follow-up
it gets the after grade and pays the follow-up's tokens too. The objective is
utility = success - lam * tokens / 100k per run (lam 0.05 = a 5-point success
gain is worth 100k tokens).

Candidate policies for finished runs, all fit only on training data:
  never, always          the fixed rules
  v0-targeted            the live /intervene rule (model groups)
  per-model              verify on models whose mean utility gain is positive
  failure-threshold      logistic P(first answer wrong | features); verify above a
                         threshold picked by inner cross-validation
  uplift-logistic        P(fixed) - P(broken) from two logistic models, minus the
                         follow-up's expected token cost
  uplift-tree            a depth-2 regression tree on the per-run utility gain

Scoring: leave-one-task-out across the training batches (a policy never sees
the task it's scored on), with a bootstrap over tasks for the gain vs the best
fixed rule; then, if --test batches are given, train on everything in
--train and score once on the held-out tasks.

    python -m learn.uplift --train scenarios/batches/cf-hard2 --test scenarios/batches/cf-hard3
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from learn.bandit import model_group
from scenarios.cf_features import FEATURE_NAMES

VERIFY_HELPS_V0 = {"helps": True, "wastes": False, "unknown": True}  # v0-targeted at its most likely draw


@dataclass
class Run:
    scenario: str
    model: str
    finished: bool
    y0: float        # success with no follow-up
    y1: float        # success with the follow-up
    t0: float        # tokens of the first turn
    t1: float        # tokens of the follow-up
    x: np.ndarray    # decision-time features


def _model_id(short: str) -> str:
    """The harness records short names (glm53f); /intervene and model_group see provider/model ids."""
    from scenarios.run_suite import MODELS, PROVIDER
    return f"{PROVIDER}/{MODELS[short][0]}" if short in MODELS else short


def load(batch_dirs: list[Path]) -> list[Run]:
    runs = []
    for d in batch_dirs:
        latest = {}
        for line in (d / "runs.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                j = json.loads(line)
                latest[j["session_key"]] = j
        for j in latest.values():
            cf = j.get("counterfactual") or {}
            before, after, feats = cf.get("before") or {}, cf.get("after") or {}, cf.get("features")
            if cf.get("action") not in ("verify", "retry") or feats is None:
                continue
            if before.get("task") is None or after.get("task") is None:
                continue  # an unverifiable grade
            runs.append(Run(scenario=j["scenario"], model=_model_id(j["model"]),
                            finished=cf["action"] == "verify", y0=float(before["task"]), y1=float(after["task"]),
                            t0=float(cf.get("tokens_first") or 0), t1=float(cf.get("tokens_follow_up") or 0),
                            x=np.array([float(feats.get(k, 0.0)) for k in FEATURE_NAMES])))
    return runs


# --- scoring ---------------------------------------------------------------------------

def score(runs: list[Run], verify: list[bool], lam: float) -> dict:
    y = np.array([r.y1 if v else r.y0 for r, v in zip(runs, verify)])
    t = np.array([r.t0 + (r.t1 if v else 0) for r, v in zip(runs, verify)])
    return {"n": len(runs), "success": float(y.mean()) if len(y) else float("nan"),
            "tokens_per_success": float(t.sum() / y.sum()) if y.sum() else float("inf"),
            "utility": float((y - lam * t / 100_000).mean()) if len(y) else float("nan"),
            "verify_rate": float(np.mean(verify)) if verify else 0.0}


def gain(r: Run, lam: float) -> float:
    """Utility of verifying this run minus not verifying it."""
    return (r.y1 - r.y0) - lam * r.t1 / 100_000


# --- learners (numpy only) ----------------------------------------------------------------

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
    """Regression tree on the per-run gain; a leaf predicts its mean."""
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
    return np.hstack([np.array([r.x for r in runs]), onehot])


def train_policies(train: list[Run], lam: float, seed: int = 0) -> dict:
    """Every candidate, fit on `train` (finished runs), as run -> verify?"""
    models = sorted({r.model for r in train})
    X = _design(train, models)
    std = Standardizer(X) if len(train) else None
    Z = std(X) if std else X
    g = np.array([gain(r, lam) for r in train])
    fix = np.array([float(r.y1 > r.y0) for r in train])
    brk = np.array([float(r.y1 < r.y0) for r in train])
    wrong = np.array([float(r.y0 == 0) for r in train])
    t1_by_model = defaultdict(list)
    for r in train:
        t1_by_model[r.model].append(r.t1)
    t1_mean = {m: float(np.mean(v)) for m, v in t1_by_model.items()}
    t1_all = float(np.mean([r.t1 for r in train])) if train else 0.0
    per_model = {m: float(np.mean([gain(r, lam) for r in train if r.model == m])) > 0 for m in models}

    w_fix, w_brk, w_wrong = (fit_logistic(Z, y) for y in (fix, brk, wrong))
    tree = fit_tree(Z, g)

    # Failure threshold: pick t by inner leave-one-task-out on the training runs.
    def thr_policy_for(t):
        return lambda r, z: predict_logistic(w_wrong, z[None])[0] > t
    scenarios = sorted({r.scenario for r in train})
    grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    inner = {t: 0.0 for t in grid}
    for s in scenarios:
        tr = [r for r in train if r.scenario != s]
        te = [r for r in train if r.scenario == s]
        if not tr or not te:
            continue
        Xtr = _design(tr, models)
        st = Standardizer(Xtr)
        w = fit_logistic(st(Xtr), np.array([float(r.y0 == 0) for r in tr]))
        p = predict_logistic(w, st(_design(te, models)))
        for t in grid:
            inner[t] += sum(gain(r, lam) for r, pi in zip(te, p) if pi > t)
    t_best = max(grid, key=lambda t: inner[t])

    def z_of(r: Run) -> np.ndarray:
        return std(_design([r], models))[0]

    return {
        "never": lambda r: False,
        "always": lambda r: True,
        "v0-targeted": lambda r: VERIFY_HELPS_V0[model_group(r.model)],
        "per-model": lambda r: per_model.get(r.model, True),
        f"failure-threshold (t={t_best})": lambda r: predict_logistic(w_wrong, z_of(r)[None])[0] > t_best,
        "uplift-logistic": lambda r: (predict_logistic(w_fix, z_of(r)[None])[0] - predict_logistic(w_brk, z_of(r)[None])[0]
                                      - lam * t1_mean.get(r.model, t1_all) / 100_000) > 0,
        "uplift-tree": lambda r: predict_tree(tree, z_of(r)) > 0,
    }


FIXED = ("never", "always", "v0-targeted")


def crossfit(runs: list[Run], lam: float) -> dict[str, list[bool]]:
    """Leave-one-task-out: each run's decision comes from policies fit without its task."""
    out = defaultdict(lambda: [False] * len(runs))
    for s in sorted({r.scenario for r in runs}):
        train = [r for r in runs if r.scenario != s]
        pols = train_policies(train, lam)
        for i, r in enumerate(runs):
            if r.scenario == s:
                for name, pi in pols.items():
                    key = name.split(" (")[0]
                    out[key][i] = bool(pi(r))
    return dict(out)


def bootstrap_gain(runs: list[Run], decisions: dict[str, list[bool]], lam: float, rounds: int = 1000, seed: int = 0) -> dict:
    """95% interval, over tasks, of each policy's utility minus the best fixed rule's (chosen per resample)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, r in enumerate(runs):
        by[r.scenario].append(i)
    tasks = sorted(by)
    diffs = defaultdict(list)
    for _ in range(rounds):
        idx = [i for s in (rng.choice(tasks) for _ in tasks) for i in by[s]]
        sub = [runs[i] for i in idx]
        util = {k: score(sub, [v[i] for i in idx], lam)["utility"] for k, v in decisions.items()}
        best_fixed = max(util[k] for k in FIXED if k in util)
        for k, u in util.items():
            diffs[k].append(u - best_fixed)
    return {k: (sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]) for k, v in diffs.items()}


def report(train: list[Run], test: list[Run] | None, lam: float, rounds: int) -> str:
    fin = [r for r in train if r.finished]
    dead = [r for r in train if not r.finished]
    lines = [f"{len(fin)} finished runs over {len({r.scenario for r in fin})} tasks; {len(dead)} dead runs; lam={lam}", ""]
    changed = [r for r in fin if r.y1 != r.y0]
    lines.append(f"Verify on finished runs: fixed {sum(r.y1 > r.y0 for r in fin)}, broke {sum(r.y1 < r.y0 for r in fin)}, "
                 f"no change {len(fin) - len(changed)}; follow-up tokens mean {np.mean([r.t1 for r in fin]):,.0f}")
    if dead:
        lines.append(f"Retry on dead runs: rescued {int(sum(r.y1 for r in dead))}/{len(dead)} "
                     f"({np.mean([r.y1 for r in dead]):.0%}), follow-up tokens mean {np.mean([r.t1 for r in dead]):,.0f}")
    lines += ["", "Leave-one-task-out (finished runs):", "",
              f"{'policy':20s} {'success':>8s} {'tok/completed':>14s} {'utility':>8s} {'verify':>7s} {'utility vs best fixed (95% CI)':>34s}"]
    dec = crossfit(fin, lam)
    ci = bootstrap_gain(fin, dec, lam, rounds)
    fixed_best = max(score(fin, dec[k], lam)["utility"] for k in FIXED)
    for k, v in dec.items():
        s = score(fin, v, lam)
        lines.append(f"{k:20s} {s['success']:8.1%} {s['tokens_per_success']:14,.0f} {s['utility']:8.3f} {s['verify_rate']:7.0%} "
                     f"{s['utility'] - fixed_best:+10.3f} ({ci[k][0]:+.3f} to {ci[k][1]:+.3f})")
    if test:
        tfin = [r for r in test if r.finished]
        pols = train_policies(fin, lam)
        lines += ["", f"Held-out tasks ({len(tfin)} finished runs over {len({r.scenario for r in tfin})} tasks), trained on all of --train:", ""]
        for name, pi in pols.items():
            s = score(tfin, [bool(pi(r)) for r in tfin], lam)
            lines.append(f"{name:32s} {s['success']:8.1%} {s['tokens_per_success']:14,.0f} {s['utility']:8.3f} {s['verify_rate']:7.0%}")
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
