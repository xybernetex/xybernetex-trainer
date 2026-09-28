import unittest

from chassis.action_space import ActionSpace
from data.trajectory_schema import ToolCallRecord, TrajectorySnapshot
from teacher import heuristic_teacher as teacher


def _snapshot(**overrides) -> TrajectorySnapshot:
    defaults = dict(run_id="r1", objective="do the thing", step_number=5, max_steps=20,
                     cost_so_far=1.0, cost_budget=10.0)
    defaults.update(overrides)
    return TrajectorySnapshot(**defaults)


def _call(tool_name="web_search", params=None, success=True, timed_out=False):
    return ToolCallRecord(tool_name=tool_name, params=params or {"q": "x"}, success=success,
                           timed_out=timed_out)


def _select(snapshot) -> str:
    return ActionSpace.decode(teacher.select_action(snapshot))


class TestHeuristicTeacher(unittest.TestCase):
    def test_destructive_tool_blocks_immediately(self):
        snap = _snapshot(tool_history=[_call(tool_name="delete_all_files")])
        self.assertEqual(_select(snap), "BLOCK_ACTION")

    def test_sensitive_tool_requests_user(self):
        snap = _snapshot(tool_history=[_call(tool_name="send_email")])
        self.assertEqual(_select(snap), "REQUEST_USER")

    def test_recorded_risk_label_decides_over_the_tool_name(self):
        shell = lambda risk: ToolCallRecord(tool_name="exec", params={"c": 1}, success=True, risk=risk)
        self.assertEqual(_select(_snapshot(tool_history=[shell("none")])), "CONTINUE")
        self.assertEqual(_select(_snapshot(tool_history=[shell("destructive")])), "BLOCK_ACTION")
        self.assertEqual(_select(_snapshot(tool_history=[shell("sensitive")])), "REQUEST_USER")

    def test_authorization_decides_whether_a_risky_call_is_stopped(self):
        """2026-09-27: BLOCK_ACTION fired on all 8 runs where an agent obeyed
        a planted instruction - and on ~100 where the user had asked for that
        exact deletion. The authorization label is what separates them."""
        def shell(risk, authorization):
            return _snapshot(tool_history=[ToolCallRecord(tool_name="exec", params={"c": 1}, success=True,
                                                          risk=risk, authorization=authorization)])
        self.assertEqual(_select(shell("destructive", "unrequested")), "BLOCK_ACTION")
        self.assertEqual(_select(shell("destructive", None)), "BLOCK_ACTION")  # unassessed: conservative
        self.assertEqual(_select(shell("destructive", "requested")), "CONTINUE")
        self.assertEqual(_select(shell("destructive", "own_artifact")), "CONTINUE")
        self.assertEqual(_select(shell("sensitive", "requested")), "CONTINUE")
        self.assertEqual(_select(shell("sensitive", "own_artifact")), "REQUEST_USER")
        self.assertEqual(_select(shell("sensitive", "unrequested")), "REQUEST_USER")

    def test_an_authorized_call_still_gets_the_other_rules(self):
        failed = ToolCallRecord(tool_name="exec", params={"c": 1}, success=False, risk="destructive",
                                authorization="requested")
        self.assertEqual(_select(_snapshot(tool_history=[failed])), "INJECT_CONTEXT")
        ok =ToolCallRecord(tool_name="exec", params={"c": 1}, success=True, risk="destructive",
                            authorization="requested")
        self.assertEqual(_select(_snapshot(tool_history=[ok], step_number=19, max_steps=20)), "TERMINATE")

    def test_safe_tool_with_no_other_issues_continues(self):
        snap = _snapshot(tool_history=[_call(tool_name="web_search")])
        self.assertEqual(_select(snap), "CONTINUE")

    def test_repeated_call_triggers_replan(self):
        snap = _snapshot(tool_history=[_call(params={"q": "a"})] * (teacher.REPEAT_THRESHOLD + 1))
        self.assertEqual(_select(snap), "REPLAN")

    def test_repeat_below_threshold_does_not_trigger_replan(self):
        snap = _snapshot(tool_history=[_call(params={"q": "a"})] * teacher.REPEAT_THRESHOLD)
        self.assertNotEqual(_select(snap), "REPLAN")

    def test_risk_tier_outranks_repeat_detection(self):
        # A repeated destructive call should still BLOCK, not REPLAN -
        # risk tier is checked first regardless of repeat count.
        snap = _snapshot(tool_history=[_call(tool_name="drop_table")] * 5)
        self.assertEqual(_select(snap), "BLOCK_ACTION")

    def test_single_failure_injects_context_not_bare_continue(self):
        snap = _snapshot(tool_history=[_call(success=True), _call(success=False)])
        self.assertEqual(_select(snap), "INJECT_CONTEXT")

    def test_consecutive_failures_escalate_to_request_user(self):
        history = [_call(success=False, params={"q": str(i)}) for i in range(teacher.ERROR_ESCALATION_THRESHOLD)]
        snap = _snapshot(tool_history=history)
        self.assertEqual(_select(snap), "REQUEST_USER")

    def test_low_step_budget_terminates(self):
        snap = _snapshot(step_number=19, max_steps=20, tool_history=[_call()])
        self.assertEqual(_select(snap), "TERMINATE")

    def test_low_cost_budget_terminates(self):
        snap = _snapshot(cost_so_far=9.5, cost_budget=10.0, tool_history=[_call()])
        self.assertEqual(_select(snap), "TERMINATE")

    def test_empty_history_does_not_crash(self):
        snap = _snapshot(tool_history=[])
        self.assertEqual(_select(snap), "CONTINUE")

    def test_zero_cost_budget_does_not_crash_or_false_trigger_termination(self):
        snap = _snapshot(cost_so_far=0.0, cost_budget=0.0, step_number=5, max_steps=20,
                          tool_history=[_call()])
        # cost_remaining defaults to 1.0 when there's no budget configured -
        # only the step budget should be able to trigger TERMINATE here.
        self.assertEqual(_select(snap), "CONTINUE")


if __name__ == "__main__":
    unittest.main()
