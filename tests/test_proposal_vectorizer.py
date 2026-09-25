from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from chassis.proposal_vectorizer import vectorize_proposal, FEATURE_SCHEMA
from chassis.state_vectorizer import vectorize
from data.trajectory_schema import ProposedToolCall, ToolCallRecord, TrajectorySnapshot
from data.replay_loader import load_proposals, load_runs


class TestProposalVectorizer(unittest.TestCase):
    def snapshot(self, **kwargs):
        return TrajectorySnapshot("run", "", 2, 50, feature_schema=FEATURE_SCHEMA, **kwargs)

    def test_pending_risk_does_not_replace_completed_risk(self):
        s = self.snapshot(tool_history=[ToolCallRecord("read", {}, True, risk="none")],
                          proposed_call=ProposedToolCall("exec", {}, "destructive"))
        v = vectorize_proposal(s)
        self.assertEqual(len(v), 148)
        self.assertEqual(v[8], 0)
        self.assertEqual(v[142], 1)
        np.testing.assert_array_equal(v[:140], vectorize(replace(s, feature_schema="xybernetex.state.v1", proposed_call=None)))

    def test_breadth_repetition_and_missing_drift_are_distinct(self):
        history = [ToolCallRecord("read", {"h": "x"}, True)] * 3
        s = self.snapshot(tool_history=history, proposed_call=ProposedToolCall("read", {"h": "x"}, "none"))
        v = vectorize_proposal(s)
        self.assertAlmostEqual(v[144], 1 / 3)
        self.assertAlmostEqual(v[145], 3 / 5)
        self.assertEqual(v[146], 0)
        self.assertEqual(v[147], 0)
        self.assertEqual(vectorize_proposal(replace(s, goal_drift_score=0.0))[147], 1)
        self.assertEqual(vectorize_proposal(replace(s, proposed_call=ProposedToolCall("new", {})))[146], 1)

    def test_legacy_and_alpha_cannot_be_confused(self):
        with self.assertRaises(ValueError):
            vectorize(self.snapshot())
        with self.assertRaises(ValueError):
            vectorize_proposal(TrajectorySnapshot("r", "", 0, 50))
        with self.assertRaises(ValueError):
            vectorize_proposal(self.snapshot(proposed_call=ToolCallRecord("read", {}, True)))

    def test_replay_separates_proposals_from_executed_steps(self):
        proposal = self.snapshot(proposed_call=ProposedToolCall("exec", {"h": "redacted"}, "none", "call"))
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.jsonl"
            log.write_text(json.dumps({"type": "tool_proposal", "runKey": "r", "snapshot": asdict(proposal)}) + "\n" +
                           json.dumps({"runKey": "r", "snapshot": asdict(replace(proposal,
                               feature_schema="xybernetex.state.v1", proposed_call=None)),
                               "decision": {"action": "CONTINUE"}}) + "\n", encoding="utf-8")
            pending = load_proposals(log)
            self.assertEqual(pending, [proposal])
            self.assertEqual(len(load_runs(log)["r"].steps), 1)
            self.assertEqual(pending[0].proposed_call.tool_call_id, "call")


if __name__ == "__main__":
    unittest.main()
