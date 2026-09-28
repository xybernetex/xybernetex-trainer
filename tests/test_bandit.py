"""The bandit's estimators and learner on synthetic runs with known truth:
logged through the same three-arm mixture as the real experiment, the
off-policy estimates must recover each policy's true success rate, and the
learner must verify only where verifying pays."""
import random
import unittest

from learn.bandit import (ARMS, Record, always, arm_probs, crossfit, evaluate, export_policy, learned_crossfit,
                          model_group, never, pooled, targeted_v0)

# Ground truth. Finished runs: base success and the lift from a verify turn.
MODELS = {
    "workers-ai/@cf/google/gemma-4-26b-a4b-it": (0.60, 0.30),   # "helps" group, verify pays
    "workers-ai/@cf/nvidia/nemotron-3-120b-a12b": (0.70, 0.00),  # "wastes" group, verify doesn't
    "anthropic/claude-x": (0.55, 0.20),                          # "unknown" group, verify pays
}
P_FINISH = 0.8
RETRY_SUCCESS = 0.5
BASE_TOKENS, VERIFY_TOKENS, RETRY_TOKENS = 100_000, 80_000, 60_000


def truth(policy) -> float:
    """Exact expected success of a policy under the simulation."""
    total = 0.0
    for model, (base, lift) in MODELS.items():
        fin = Record("x", "x", model, True, False, 5, "none", False, 0, 0.0)
        dead = Record("x", "x", model, False, True, 0, "none", False, 0, 0.0)
        p_verify = policy(fin).get("verify", 0.0)
        p_retry = policy(dead).get("retry", 0.0)
        total += P_FINISH * (base + p_verify * lift) + (1 - P_FINISH) * p_retry * RETRY_SUCCESS
    return total / len(MODELS)


def simulate(n_scenarios=60, reps=2, seed=1) -> list:
    rng = random.Random(seed)
    records = []
    for s in range(n_scenarios):
        for model, (base, lift) in MODELS.items():
            for arm in ARMS:
                for _ in range(reps):
                    finished = rng.random() < P_FINISH
                    r = Record(arm, f"scn{s}", model, finished, not finished, 5 if finished else 0, "none", False, 0, 0.0)
                    probs = arm_probs(arm, r)
                    u, acc = rng.random(), 0.0
                    for a, p in probs.items():
                        acc += p
                        if u < acc:
                            r.action = a
                            break
                    if finished:
                        r.success = rng.random() < base + (lift if r.action == "verify" else 0)
                        r.tokens = BASE_TOKENS + (VERIFY_TOKENS if r.action == "verify" else 0)
                    else:
                        r.success = r.action == "retry" and rng.random() < RETRY_SUCCESS
                        r.tokens = 40_000 + (RETRY_TOKENS if r.action == "retry" else 0)
                    r.behavior = pooled(r)
                    records.append(r)
    return records


class BanditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = simulate()
        cls.folds, cls.models = crossfit(cls.records)

    def test_propensities_mirror_the_live_rule(self):
        fin = Record("x", "x", "workers-ai/@cf/google/gemma-4-26b-a4b-it", True, False, 3, "none", False, 0, 0.0)
        self.assertEqual(model_group(fin.model), "helps")
        self.assertAlmostEqual(arm_probs("targeted", fin)["verify"], 0.9)
        self.assertAlmostEqual(pooled(fin)["verify"], (0 + 1 + 0.9) / 3)
        no_work = Record("x", "x", "m", True, False, 0, "none", False, 0, 0.0)
        self.assertEqual(pooled(no_work), {"none": 1.0, "verify": 0.0})

    def test_estimators_recover_the_truth(self):
        for name, policy in (("never", never), ("always", always), ("targeted", targeted_v0)):
            est = evaluate(self.records, policy, self.models, self.folds)
            self.assertAlmostEqual(est["dr_success"], truth(policy), delta=0.04, msg=f"{name} DR")
            self.assertAlmostEqual(est["snips_success"], truth(policy), delta=0.05, msg=f"{name} SNIPS")

    def test_learner_verifies_only_where_it_pays(self):
        table = export_policy(self.records, lam=0.05, epsilon=0.0)["models"]
        self.assertEqual(table["workers-ai/@cf/google/gemma-4-26b-a4b-it"]["verify"], 1.0)
        self.assertEqual(table["anthropic/claude-x"]["verify"], 1.0)
        self.assertEqual(table["workers-ai/@cf/nvidia/nemotron-3-120b-a12b"]["verify"], 0.0)
        self.assertEqual(table["anthropic/claude-x"]["retry"], 1.0)

    def test_the_learned_policy_beats_the_best_fixed_rule_out_of_sample(self):
        learned = learned_crossfit(self.records, self.folds, self.models, lam=0.05)
        est = {name: evaluate(self.records, p, self.models, self.folds)
               for name, p in (("never", never), ("always", always), ("targeted", targeted_v0), ("learned", learned))}
        # Same success as always-verify (it keeps every paying verify)...
        self.assertGreater(est["learned"]["dr_success"], est["always"]["dr_success"] - 0.03)
        # ...for fewer tokens per completed task (it drops the wasted ones).
        self.assertLess(est["learned"]["dr_tokens_per_success"], est["always"]["dr_tokens_per_success"])

    def test_exploration_keeps_every_action_possible(self):
        table = export_policy(self.records, lam=0.05, epsilon=0.2)["models"]
        self.assertAlmostEqual(table["workers-ai/@cf/nvidia/nemotron-3-120b-a12b"]["verify"], 0.1)
        self.assertAlmostEqual(table["anthropic/claude-x"]["verify"], 0.9)


if __name__ == "__main__":
    unittest.main()
