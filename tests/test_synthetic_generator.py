import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chassis.action_space import ActionSpace
from chassis.state_vectorizer import TRAINING_STATE_DIM
from data import synthetic_generator as gen

TERMINATE = ActionSpace.encode("TERMINATE")


class TestSyntheticGenerator(unittest.TestCase):
    def test_output_shapes_and_dtypes(self):
        states, actions = gen.generate_episodes(num_episodes=20)
        self.assertEqual(states.shape[1], TRAINING_STATE_DIM)
        self.assertEqual(states.shape[0], actions.shape[0])
        self.assertEqual(states.dtype, np.float32)
        self.assertEqual(actions.dtype, np.int64)

    def test_same_seed_is_reproducible(self):
        states_a, actions_a = gen.generate_episodes(num_episodes=20, seed=7)
        states_b, actions_b = gen.generate_episodes(num_episodes=20, seed=7)
        np.testing.assert_array_equal(states_a, states_b)
        np.testing.assert_array_equal(actions_a, actions_b)

    def test_all_actions_appear_with_enough_episodes(self):
        _, actions = gen.generate_episodes(num_episodes=300)
        seen = {ActionSpace.decode(a) for a in np.unique(actions)}
        for expected in ("CONTINUE", "REPLAN", "BLOCK_ACTION", "REQUEST_USER", "TERMINATE"):
            self.assertIn(expected, seen, f"{expected} never appeared in 300 synthetic episodes")

    def test_only_budget_exhaustion_episodes_terminate(self):
        """Regression test for a code-review finding (2026-09-25): the cost
        budget used to run out at the end of EVERY episode, so every category
        ended in TERMINATE and the model learned "late in the episode =>
        TERMINATE" instead of "budget is actually low => TERMINATE"."""
        rng = random.Random(0)
        for category in ("normal", "stuck_loop", "error_recovery", "risky_tool"):
            for _ in range(100):
                _states, actions = gen.generate_episode(category, rng)
                self.assertNotIn(TERMINATE, actions, category)

    def test_budget_exhaustion_episodes_do_terminate(self):
        rng = random.Random(0)
        for _ in range(100):
            _states, actions = gen.generate_episode("budget_exhaustion", rng)
            self.assertEqual(actions[-1], TERMINATE)

    def test_trigger_positions_vary(self):
        """The other half of the same finding: triggers used to sit at a fixed
        position, so the dataset couldn't tell "learned the causal feature"
        apart from "learned the position"."""
        rng = random.Random(0)
        block_positions = set()
        for _ in range(100):
            _states, actions = gen.generate_episode("risky_tool", rng)
            block_positions.update(i for i, a in enumerate(actions)
                                   if ActionSpace.decode(a) in ("BLOCK_ACTION", "REQUEST_USER"))
        self.assertGreater(len(block_positions), 10)

    def test_error_recovery_streak_length_varies_and_reaches_saturation(self):
        """Regression test for a live-data finding (2026-09-25, full-run1 +
        offline ablation - see the module docstring): the streak was fixed at
        exactly 3, so consecutive_error_count's normalization cap of 5 was
        never reached in training, and the model was confidently wrong on
        real streaks of 6-8, worse the longer the streak ran."""
        rng = random.Random(0)
        lengths = set()
        for _ in range(200):
            history = gen._build_episode_history("error_recovery", rng, length=20)
            best = cur = 0
            for call in history:
                cur = cur + 1 if not call.success else 0
                best = max(best, cur)
            lengths.add(best)
        self.assertTrue(lengths & {2, 3, 4}, f"no sub-cap streak length seen: {lengths}")
        self.assertTrue(lengths & {5, 6, 7}, f"no cap-saturating streak length (>=5) seen: {lengths}")

    def test_budget_exhaustion_continues_with_prior_terminate_in_history(self):
        """Regression test for the other live-data finding: episodes used to
        end at (or one step after) their own trigger, so the model never
        trained on its own TERMINATE recommendation already sitting in
        recent_actions - exactly what real observe-only deployment produces,
        since nothing there stops a run. An offline ablation against the
        deployed checkpoint found this was a hard, instantly-saturating
        switch (P(TERMINATE) 0.56->0.94 on one prior entry) rather than
        something the model had ever been trained against."""
        rng = random.Random(0)
        saw_prior_terminate = False
        for _ in range(100):
            steps = gen.episode_snapshots("budget_exhaustion", rng)
            for snapshot, action in steps:
                if ActionSpace.decode(action) == "TERMINATE" and "TERMINATE" in snapshot.recent_actions:
                    saw_prior_terminate = True
                    break
            if saw_prior_terminate:
                break
        self.assertTrue(saw_prior_terminate, "no budget_exhaustion step ever had a prior TERMINATE in history")

    def test_non_exhaustion_categories_sometimes_reach_high_usage_without_terminating(self):
        """The other half of the density-balance fix: non-exhaustion
        categories used to cap at ~33% budget used across their own real
        length, so the model saw almost no genuinely-CONTINUE examples in the
        65-90% zone - only budget_exhaustion's own thin pre-trigger ramp did,
        and that was too sparse to offset the extra TERMINATE mass the
        continuation fix above adds there. This must never cross into an
        actual TERMINATE for a non-exhaustion category (the original
        2026-09-25 code-review fix's invariant)."""
        rng = random.Random(0)
        reached_high_usage = False
        for category in ("normal", "stuck_loop", "error_recovery", "risky_tool"):
            for _ in range(100):
                for snapshot, action in gen.episode_snapshots(category, rng):
                    used = max(snapshot.step_number / max(snapshot.max_steps, 1),
                              snapshot.cost_so_far / max(snapshot.cost_budget, 1e-9))
                    if used >= 0.65:
                        reached_high_usage = True
                    self.assertNotEqual(ActionSpace.decode(action), "TERMINATE",
                                        f"{category} terminated at {used:.2f} used")
        self.assertTrue(reached_high_usage, "no non-exhaustion episode ever reached 65% budget used")

    def test_every_risk_tier_and_authorization_pair_is_covered(self):
        """The teacher's risky-call decision now depends on the label, so the
        model has to see each tier x label pair to learn it."""
        rng = random.Random(0)
        seen = set()
        for _ in range(300):
            for snapshot, _action in gen.episode_snapshots("risky_tool", rng):
                last = snapshot.tool_history[-1]
                tier = "destructive" if last.tool_name in gen._DESTRUCTIVE_TOOLS else (
                    "sensitive" if last.tool_name in gen._SENSITIVE_TOOLS else None)
                if tier:
                    seen.add((tier, last.authorization))
        want = {(t, a) for t in ("destructive", "sensitive") for a in ("requested", "own_artifact", "unrequested", None)}
        self.assertEqual(want - seen, set())

    def test_requested_risky_calls_that_keep_failing_are_trained_as_error_streaks(self):
        """Regression for the 2026-09-27 live miss: a requested delete that
        failed twice drew BLOCK_ACTION (0.64) from the v3 policy, because no
        synthetic risky call had ever failed. Both escalation steps must now
        appear for requested destructive calls."""
        rng = random.Random(2)
        seen = set()
        for _ in range(400):
            for snapshot, action in gen.episode_snapshots("risky_tool", rng):
                last = snapshot.tool_history[-1]
                if last.tool_name in gen._DESTRUCTIVE_TOOLS and last.authorization == "requested" and not last.success:
                    streak = 0
                    for call in reversed(snapshot.tool_history):
                        if call.success:
                            break
                        streak += 1
                    seen.add((min(streak, 2), ActionSpace.decode(action)))
        self.assertIn((1, "INJECT_CONTEXT"), seen)
        self.assertIn((2, "REQUEST_USER"), seen)
        self.assertFalse(any(a == "BLOCK_ACTION" for _, a in seen), seen)

    def test_requested_risky_calls_are_not_blocked(self):
        rng = random.Random(1)
        for _ in range(200):
            for snapshot, action in gen.episode_snapshots("risky_tool", rng):
                last = snapshot.tool_history[-1]
                if last.tool_name in gen._DESTRUCTIVE_TOOLS and last.authorization == "requested":
                    self.assertNotEqual(ActionSpace.decode(action), "BLOCK_ACTION")

    def test_actions_are_valid_indices(self):
        _, actions = gen.generate_episodes(num_episodes=20)
        self.assertTrue(np.all(actions >= 0))
        self.assertTrue(np.all(actions < ActionSpace.N))

    def test_generate_and_save_writes_loadable_npz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "rollouts.npz")
            gen.generate_and_save(num_episodes=5, output_path=out_path)
            # np.load() keeps the zip file handle open until closed - on
            # Windows that blocks the tempdir cleanup below unless this is
            # a context manager (or .close() is called explicitly).
            with np.load(out_path) as data:
                self.assertEqual(set(data.files), {"states", "actions"})
                self.assertEqual(data["states"].shape[1], TRAINING_STATE_DIM)
                self.assertEqual(data["states"].shape[0], data["actions"].shape[0])


if __name__ == "__main__":
    unittest.main()
