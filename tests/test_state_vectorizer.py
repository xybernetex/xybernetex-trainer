import unittest

import numpy as np

from chassis import state_vectorizer as sv
from chassis.action_space import ActionSpace
from data.trajectory_schema import ToolCallRecord, TrajectorySnapshot


def _snapshot(**overrides) -> TrajectorySnapshot:
    defaults = dict(run_id="r1", objective="do the thing", step_number=5, max_steps=20)
    defaults.update(overrides)
    return TrajectorySnapshot(**defaults)


def _call(tool_name="web_search", params=None, success=True, error=None, timed_out=False):
    return ToolCallRecord(tool_name=tool_name, params=params or {"q": "x"}, success=success,
                           error=error, timed_out=timed_out)


class TestStateVectorizer(unittest.TestCase):
    def test_output_shape_and_dtype(self):
        vec = sv.vectorize(_snapshot())
        self.assertEqual(vec.shape, (sv.STATE_DIM,))
        self.assertEqual(vec.dtype, np.float32)

    def test_step_budget_features(self):
        vec = sv.vectorize(_snapshot(step_number=5, max_steps=20))
        self.assertAlmostEqual(vec[0], 5 / 20)
        self.assertAlmostEqual(vec[1], 15 / 20)

    def test_repeat_detection(self):
        same_twice = _snapshot(tool_history=[_call(params={"q": "a"}), _call(params={"q": "a"})])
        different = _snapshot(tool_history=[_call(params={"q": "a"}), _call(params={"q": "b"})])
        self.assertGreater(sv.vectorize(same_twice)[3], 0.0)
        self.assertEqual(sv.vectorize(different)[3], 0.0)

    def test_consecutive_repeat_count_scales_with_streak(self):
        one_repeat = _snapshot(tool_history=[_call(params={"q": "a"}), _call(params={"q": "a"})])
        three_repeats = _snapshot(tool_history=[_call(params={"q": "a"})] * 4)
        self.assertLess(sv.vectorize(one_repeat)[3], sv.vectorize(three_repeats)[3])
        self.assertGreater(sv.vectorize(three_repeats)[3], 0.0)

    def test_recent_call_outcome_rates(self):
        history = [_call(success=True), _call(success=False, timed_out=True), _call(success=False)]
        vec = sv.vectorize(_snapshot(tool_history=history))
        self.assertAlmostEqual(vec[4], 1 / 3)  # success_rate
        self.assertAlmostEqual(vec[5], 1 / 3)  # timeout_rate
        self.assertAlmostEqual(vec[6], 1 / 3)  # error_rate (failed, not timed out)

    def test_consecutive_error_count(self):
        history = [_call(success=True), _call(success=False), _call(success=False)]
        vec = sv.vectorize(_snapshot(tool_history=history))
        self.assertAlmostEqual(vec[7], 2 / sv._MAX_CONSECUTIVE_ERRORS)

    def test_risk_tiers_get_separate_bits(self):
        """Regression test: the two tiers used to share one bit, so the policy
        couldn't tell BLOCK_ACTION cases from REQUEST_USER ones."""
        destructive = sv.vectorize(_snapshot(tool_history=[_call(tool_name="shell_exec")]))
        sensitive = sv.vectorize(_snapshot(tool_history=[_call(tool_name="send_email")]))
        safe = sv.vectorize(_snapshot(tool_history=[_call(tool_name="web_search")]))
        self.assertEqual((destructive[8], destructive[2]), (1.0, 0.0))
        self.assertEqual((sensitive[8], sensitive[2]), (0.0, 1.0))
        self.assertEqual((safe[8], safe[2]), (0.0, 0.0))

    def test_risk_tier_matches_whole_tokens_not_substrings(self):
        # Each of these was misclassified by the old substring matcher.
        for benign in ("google_drive_file_retrieval", "format_document", "confirm_order",
                       "platform_status", "dropdown_select", "postgres_query"):
            self.assertIsNone(sv.risk_tier(benign), benign)

    def test_risk_tier_still_catches_real_risky_names(self):
        for name, tier in (("delete_file", "destructive"), ("drop_table", "destructive"),
                           ("shell_exec", "destructive"), ("run_command", "destructive"),
                           ("rm", "destructive"), ("deleteFile", "destructive"),
                           ("send_email", "sensitive"), ("publish_post", "sensitive"),
                           ("transfer_funds", "sensitive"), ("sendEmail", "sensitive")):
            self.assertEqual(sv.risk_tier(name), tier, name)

    def test_risk_tier_prefers_destructive_when_both_match(self):
        self.assertEqual(sv.risk_tier("delete_and_publish"), "destructive")

    def test_recorded_risk_label_overrides_the_tool_name(self):
        """A shell tool's name says nothing about what it ran - OpenClaw's exec
        runs `python test.py` and `Remove-Item` alike - so a label recorded
        from the actual command wins, including an explicit "none"."""
        def bits(call):
            vec = sv.vectorize(_snapshot(tool_history=[call]))
            return (vec[8], vec[2])

        exec_call = lambda risk: ToolCallRecord(tool_name="exec", params={}, success=True, risk=risk)
        self.assertEqual(bits(exec_call(None)), (1.0, 0.0))  # no label: judged by name
        self.assertEqual(bits(exec_call("none")), (0.0, 0.0))
        self.assertEqual(bits(exec_call("sensitive")), (0.0, 1.0))
        read_call = ToolCallRecord(tool_name="read", params={}, success=True, risk="destructive")
        self.assertEqual(bits(read_call), (1.0, 0.0))

    def test_unknown_risk_label_is_rejected(self):
        call = ToolCallRecord(tool_name="exec", params={}, success=True, risk="scary")
        with self.assertRaises(ValueError):
            sv.vectorize(_snapshot(tool_history=[call]))

    def test_cost_budget_features(self):
        vec = sv.vectorize(_snapshot(cost_so_far=3.0, cost_budget=10.0))
        self.assertAlmostEqual(vec[9], 0.3)
        self.assertAlmostEqual(vec[10], 0.7)

    def test_cost_budget_zero_does_not_crash(self):
        vec = sv.vectorize(_snapshot(cost_so_far=0.0, cost_budget=0.0))
        self.assertEqual(vec[9], 0.0)
        self.assertEqual(vec[10], 0.0)

    def test_goal_drift_defaults_to_zero_when_not_computed(self):
        vec = sv.vectorize(_snapshot(goal_drift_score=None))
        self.assertEqual(vec[11], 0.0)

    def test_goal_drift_passthrough_when_computed(self):
        vec = sv.vectorize(_snapshot(goal_drift_score=0.42))
        self.assertAlmostEqual(vec[11], 0.42)

    def test_action_history_right_aligned_and_padded(self):
        vec = sv.vectorize(_snapshot(recent_actions=["CONTINUE"]))
        history_block = vec[sv._STRUCT_DIM:].reshape(sv.HISTORY_LEN, sv.ACTION_COUNT)
        # only the last row should be non-zero, at CONTINUE's index
        self.assertTrue(np.all(history_block[:-1] == 0))
        expected_row = np.zeros(sv.ACTION_COUNT, dtype=np.float32)
        expected_row[ActionSpace.encode("CONTINUE")] = 1.0
        np.testing.assert_array_equal(history_block[-1], expected_row)

    def test_action_history_truncates_to_history_len(self):
        actions = ["CONTINUE"] * (sv.HISTORY_LEN + 5)
        vec = sv.vectorize(_snapshot(recent_actions=actions))
        history_block = vec[sv._STRUCT_DIM:].reshape(sv.HISTORY_LEN, sv.ACTION_COUNT)
        self.assertTrue(np.all(history_block.sum(axis=1) == 1.0))

    def test_no_nans_or_infs_on_empty_snapshot(self):
        vec = sv.vectorize(_snapshot())
        self.assertFalse(np.any(np.isnan(vec)))
        self.assertFalse(np.any(np.isinf(vec)))


if __name__ == "__main__":
    unittest.main()
