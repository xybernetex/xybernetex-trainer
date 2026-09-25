from copy import deepcopy
import unittest

from scenarios.proposal_report import audit


class TestProposalReport(unittest.TestCase):
    def events(self):
        snapshot = {"run_id": "r", "step_number": 0, "max_steps": 50,
                    "feature_schema": "xybernetex.state.v2", "tool_history": [],
                    "proposed_call": {"tool_name": "exec", "params": {"h": "digest"},
                                      "risk": "destructive", "tool_call_id": "c"}}
        return [{"type": "tool_proposal", "runKey": "r", "toolCallId": "c", "snapshot": snapshot},
                {"type": "tool_completed", "runKey": "r", "toolCallId": "c",
                 "record": {"tool_name": "exec", "params": {"h": "digest"}, "success": True}}]

    def test_completion_does_not_fabricate_training_label(self):
        report = audit(self.events())
        self.assertEqual(report["statuses"], {"completed": 1})
        self.assertIsNone(report["rows"][0]["target_action"])
        self.assertEqual(report["rows"][0]["snapshot"]["tool_history"], [])

    def test_missing_outcome_and_other_run_do_not_imply_completion(self):
        entries = self.events()
        entries[1]["runKey"] = "other"
        self.assertEqual(audit(entries)["statuses"], {"pending_or_missing": 1})
        self.assertEqual(audit(entries)["unmatched_completions"], 1)
        entries[0].pop("toolCallId")
        self.assertEqual(audit(entries)["statuses"], {"invalid": 1})

    def test_changed_call_and_reversed_order_are_invalid(self):
        entries = self.events()
        entries[1]["record"]["params"]["h"] = "changed"
        row = audit(entries[::-1])["rows"][0]
        self.assertIn("outcome_precedes_proposal", row["problems"])
        self.assertIn("call_changed_or_mismatched", row["problems"])

    def test_duplicate_id_is_not_arbitrarily_matched(self):
        entries = self.events()
        entries.insert(1, deepcopy(entries[0]))
        self.assertEqual(audit(entries)["statuses"], {"invalid": 2})

    def test_explicit_block_is_not_a_general_teacher_label(self):
        entries = self.events()
        gate = {"type": "tool_gate", "runKey": "r", "toolCallId": "c", "toolName": "exec",
                "paramsHash": "digest", "ruleId": "test-only", "action": "BLOCK_ACTION", "enforced": True}
        self.assertEqual(audit([entries[0], gate])["statuses"], {"blocked": 1})
        self.assertEqual(audit([entries[0], gate, entries[1]])["statuses"], {"invalid": 1})


if __name__ == "__main__":
    unittest.main()
