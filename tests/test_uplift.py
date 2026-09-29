"""learn.uplift on synthetic counterfactual runs with known truth: exact
scoring, the numpy learners, and that learners using decision-time features
pick the right follow-up per run and beat every fixed rule when each
follow-up pays in a different, visible niche - and claim nothing when no
follow-up ever pays."""
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from learn import uplift
from scenarios.cf_features import FEATURE_NAMES, features, split_turns, tokens

MODELS = ["workers-ai/@cf/google/gemma-4-26b-a4b-it", "workers-ai/@cf/nvidia/nemotron-3-120b-a12b"]
COST = {"verify-same": 40_000, "test-fresh": 60_000, "verify-strong": 150_000}


def synth(n_tasks=12, per_task=40, seed=1, pays=True) -> list[uplift.Run]:
    """Wrong first answers get likelier with failed calls. The test-writing
    check fixes most wrong answers from runs that never claimed to check;
    the same-session check fixes some from runs that hedged; the strong model
    fixes more of any kind but costs far more."""
    rng = random.Random(seed)
    runs = []
    for t in range(n_tasks):
        for _ in range(per_task):
            f = {k: 0.0 for k in FEATURE_NAMES}
            f["failed_calls"] = rng.randint(0, 6)
            f["hedges"] = float(rng.random() < 0.4)
            f["claims_checked"] = float(rng.random() < 0.5)
            wrong = rng.random() < 0.15 + 0.1 * f["failed_calls"]
            y0 = 0.0 if wrong else 1.0
            fixes = {"test-fresh": pays and wrong and not f["claims_checked"] and rng.random() < 0.8,
                     "verify-same": pays and wrong and f["hedges"] and rng.random() < 0.5,
                     "verify-strong": pays and wrong and rng.random() < 0.6}
            runs.append(uplift.Run(scenario=f"t{t}", model=rng.choice(MODELS), kind="finished", y0=y0, t0=100_000,
                                   x=np.array([f[k] for k in FEATURE_NAMES]),
                                   arms={a: (1.0 if (y0 or fixes[a]) else 0.0, COST[a]) for a in COST}))
    return runs


class ScoringTest(unittest.TestCase):
    def test_exact_scoring_uses_each_runs_own_outcome_under_the_chosen_action(self):
        z = np.zeros(len(FEATURE_NAMES))
        runs = [uplift.Run("a", "m", "finished", 0, 100, z, {"v": (1, 50)}),
                uplift.Run("a", "m", "finished", 1, 100, z, {"v": (1, 50)})]
        never = uplift.score(runs, ["none", "none"], lam=0.0)
        both = uplift.score(runs, ["v", "v"], lam=0.0)
        self.assertEqual((never["success"], never["tokens_per_success"]), (0.5, 200.0))
        self.assertEqual((both["success"], both["tokens_per_success"]), (1.0, 150.0))

    def test_logistic_and_tree_recover_simple_structure(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 2))
        y = (X[:, 0] > 0).astype(float)
        w = uplift.fit_logistic(X, y)
        self.assertGreater(((uplift.predict_logistic(w, X) > 0.5) == (y == 1)).mean(), 0.95)
        tree = uplift.fit_tree(X, np.where(X[:, 1] > 0.5, 1.0, -1.0))
        self.assertEqual(uplift.predict_tree(tree, np.array([0.0, 2.0])), 1.0)
        self.assertEqual(uplift.predict_tree(tree, np.array([0.0, -2.0])), -1.0)


class LearningTest(unittest.TestCase):
    def test_learners_pick_the_right_follow_up_per_run_and_beat_every_fixed_rule(self):
        runs = synth()
        dec = uplift.crossfit(runs, lam=0.05)
        util = {k: uplift.score(runs, v, 0.05)["utility"] for k, v in dec.items()}
        best_fixed = max(util[k] for k in uplift.fixed_names(runs))
        for k in ("uplift-tree", "uplift-logistic"):
            self.assertGreater(util[k], best_fixed + 0.015, util)
        # It uses the test-writing check mostly where the run never claimed to check.
        i_claims = FEATURE_NAMES.index("claims_checked")
        test_picks = [r.x[i_claims] for r, a in zip(runs, dec["uplift-tree"]) if a == "test-fresh"]
        self.assertLess(np.mean(test_picks), 0.2)
        # And out of sample, on new tasks.
        test = synth(n_tasks=6, seed=9)
        pols = uplift.train_policies(runs, 0.05)
        held = {k: uplift.score(test, [p(r) for r in test], 0.05)["utility"] for k, p in pols.items()}
        self.assertGreater(held["uplift-tree"], max(held[k] for k in uplift.fixed_names(runs)))

    def test_no_learner_claims_a_win_when_no_follow_up_ever_pays(self):
        runs = synth(pays=False)
        dec = uplift.crossfit(runs, lam=0.05)
        util = {k: uplift.score(runs, v, 0.05)["utility"] for k, v in dec.items()}
        self.assertEqual(max(util, key=util.get), "never")  # every follow-up only costs tokens
        ci = uplift.bootstrap_gain(runs, dec, 0.05, rounds=200)
        for k in ("uplift-tree", "uplift-logistic", "per-model", "failure-threshold"):
            self.assertLessEqual(ci[k][0], 0.0)  # no learned policy is confidently better


class CounterfactualDataTest(unittest.TestCase):
    def test_transcripts_split_at_the_follow_up_and_features_are_counts(self):
        msgs = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": [{"type": "toolCall", "name": "write", "arguments": {"p": 1}}],
             "usage": {"totalTokens": 1000}, "timestamp": 0, "stopReason": "toolUse"},
            {"role": "toolResult", "isError": True, "content": [], "timestamp": 30_000},
            {"role": "assistant", "content": [{"type": "toolCall", "name": "write", "arguments": {"p": 1}}],
             "usage": {"totalTokens": 500}, "timestamp": 60_000},
            {"role": "toolResult", "isError": False, "content": []},
            {"role": "assistant", "content": [{"type": "text", "text": "Done, it should work."}],
             "usage": {"totalTokens": 200}, "timestamp": 120_000, "stopReason": "stop"},
            {"role": "user", "content": "[verify]"},
            {"role": "assistant", "content": [{"type": "text", "text": "Checked."}], "usage": {"totalTokens": 300}},
        ]
        first, follow = split_turns(msgs)
        self.assertEqual((len(first), len(follow)), (6, 2))
        self.assertEqual((tokens(first), tokens(follow)), (1700, 300))
        f = features(first)
        self.assertEqual((f["tool_calls"], f["failed_calls"], f["writes"], f["max_repeat"]), (2, 1, 2, 2))
        self.assertEqual((f["hedges"], f["claims_checked"], f["ended_stop"], f["minutes"]), (1, 0, 1, 2.0))
        self.assertTrue(all(isinstance(v, (int, float)) for v in f.values()))

    def test_load_reads_every_branch_and_skips_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as d:
            feats = {k: 1 for k in FEATURE_NAMES}
            ok = {"task": True, "tokens": 100}
            rows = [
                {"session_key": "k1", "scenario": "hx-a", "model": "gemma4",
                 "counterfactual": {"kind": "finished", "before": {"task": False}, "features": feats, "tokens_first": 900,
                                    "branches": {"verify-same": ok, "test-fresh": {"task": False, "tokens": 50}}}},
                {"session_key": "k2", "scenario": "hx-a", "model": "gemma4",
                 "counterfactual": {"kind": None, "before": {"task": False}, "note": "timed out", "branches": {}}},
                {"session_key": "k3", "scenario": "hx-b", "model": "nemo3",
                 "counterfactual": {"kind": "died", "before": {"task": False}, "features": feats,
                                    "branches": {"retry-same": ok, "retry-fresh": {"task": None, "error": "x"}}}},
            ]
            (Path(d) / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            runs = uplift.load([Path(d)])
        self.assertEqual(len(runs), 1)  # no follow-ups, and a branch without a grade, are left out
        r = runs[0]
        self.assertEqual((r.y0, r.arms["verify-same"], r.arms["test-fresh"]), (0.0, (1.0, 100.0), (0.0, 50.0)))
        self.assertEqual(r.outcome("verify-same"), (1.0, 1000.0))
        self.assertTrue(r.model.endswith("gemma-4-26b-a4b-it"))


if __name__ == "__main__":
    unittest.main()


class ConstantLabelsTest(unittest.TestCase):
    """A fold where an action never fixed a run used to crash with a singular Hessian."""

    def test_constant_labels_give_a_finite_base_rate(self):
        import numpy as np
        from learn.uplift import fit_logistic, predict_logistic
        X = np.random.default_rng(0).normal(size=(12, 4))
        for y in (np.zeros(12), np.ones(12)):
            w = fit_logistic(X, y)
            p = predict_logistic(w, X)
            self.assertTrue(np.all(np.isfinite(w)))
            self.assertTrue(np.allclose(p, p[0]))
            self.assertTrue(0 < p[0] < 1)
