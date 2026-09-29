"""learn.uplift on synthetic counterfactual runs with known truth: exact
scoring, the numpy learners, and that a learner using decision-time features
beats every fixed rule when verification only pays on runs those features
can spot - and doesn't when verification never pays."""
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from learn import uplift
from scenarios.cf_features import FEATURE_NAMES, features, split_turns, tokens

MODELS = ["workers-ai/@cf/google/gemma-4-26b-a4b-it", "workers-ai/@cf/nvidia/nemotron-3-120b-a12b"]


def synth(n_tasks=12, per_task=40, seed=1, verify_pays=True) -> list[uplift.Run]:
    """First answers are wrong more often when failed_calls is high; verify fixes
    most wrong answers only when the run hedged (a visible signal) and costs
    60k tokens every time."""
    rng = random.Random(seed)
    runs = []
    for t in range(n_tasks):
        for _ in range(per_task):
            f = {k: 0.0 for k in FEATURE_NAMES}
            f["failed_calls"] = rng.randint(0, 6)
            f["hedges"] = float(rng.random() < 0.4)
            f["tool_calls"] = rng.randint(3, 30)
            wrong = rng.random() < 0.15 + 0.1 * f["failed_calls"]
            y0 = 0.0 if wrong else 1.0
            fixes = verify_pays and wrong and f["hedges"] and rng.random() < 0.8
            y1 = 1.0 if (y0 or fixes) else 0.0
            runs.append(uplift.Run(scenario=f"t{t}", model=rng.choice(MODELS), finished=True, y0=y0, y1=y1,
                                   t0=100_000, t1=60_000, x=np.array([f[k] for k in FEATURE_NAMES])))
    return runs


class ScoringTest(unittest.TestCase):
    def test_exact_scoring_uses_each_runs_own_outcome_under_the_chosen_action(self):
        runs = [uplift.Run("a", "m", True, 0, 1, 100, 50, np.zeros(len(FEATURE_NAMES))),
                uplift.Run("a", "m", True, 1, 1, 100, 50, np.zeros(len(FEATURE_NAMES)))]
        never = uplift.score(runs, [False, False], lam=0.0)
        both = uplift.score(runs, [True, True], lam=0.0)
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
    def test_feature_learners_beat_every_fixed_rule_when_the_signal_is_visible(self):
        runs = synth()
        dec = uplift.crossfit(runs, lam=0.05)
        util = {k: uplift.score(runs, v, 0.05)["utility"] for k, v in dec.items()}
        best_fixed = max(util[k] for k in uplift.FIXED)
        # The best any policy can do from the features: verify exactly the runs that hedged
        # (skipping the rest saves 0.03 utility each; about 60% of runs, so at most ~+0.018).
        oracle = uplift.score(runs, [bool(r.x[FEATURE_NAMES.index("hedges")]) for r in runs], 0.05)["utility"]
        for k in ("uplift-tree", "uplift-logistic"):
            self.assertGreater(util[k], best_fixed + 0.01, util)
            self.assertGreater(util[k], oracle - 0.005, (k, util[k], oracle))
        # Out of sample on new tasks too.
        test = synth(n_tasks=6, seed=9)
        pols = uplift.train_policies(runs, 0.05)
        held = {k.split(" (")[0]: uplift.score(test, [bool(p(r)) for r in test], 0.05)["utility"] for k, p in pols.items()}
        self.assertGreater(held["uplift-tree"], max(held[k] for k in uplift.FIXED))

    def test_no_learner_claims_a_win_when_verification_never_pays(self):
        runs = synth(verify_pays=False)
        dec = uplift.crossfit(runs, lam=0.05)
        util = {k: uplift.score(runs, v, 0.05)["utility"] for k, v in dec.items()}
        self.assertEqual(max(util, key=util.get), "never")  # verifying only ever costs tokens
        ci = uplift.bootstrap_gain(runs, dec, 0.05, rounds=200)
        for k in ("uplift-tree", "uplift-logistic", "per-model"):
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

    def test_load_reads_the_harness_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            feats = {k: 1 for k in FEATURE_NAMES}
            rows = [
                {"session_key": "k1", "scenario": "hx-a", "model": "gemma4",
                 "counterfactual": {"action": "verify", "before": {"task": False}, "after": {"task": True},
                                    "features": feats, "tokens_first": 900, "tokens_follow_up": 100}},
                {"session_key": "k2", "scenario": "hx-a", "model": "gemma4",
                 "counterfactual": {"action": None, "before": {"task": False}, "note": "timed out"}},
                {"session_key": "k3", "scenario": "hx-b", "model": "nemo3",
                 "counterfactual": {"action": "retry", "before": {"task": False}, "after": {"task": None},
                                    "features": feats}},
            ]
            (Path(d) / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            runs = uplift.load([Path(d)])
        self.assertEqual(len(runs), 1)  # no follow-up, and an unverifiable grade, are left out
        self.assertEqual((runs[0].y0, runs[0].y1, runs[0].t1), (0.0, 1.0, 100.0))
        self.assertTrue(runs[0].model.endswith("gemma-4-26b-a4b-it"))


if __name__ == "__main__":
    unittest.main()
