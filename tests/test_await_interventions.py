"""run_suite's wait for plugin-started follow-ups: it reads only this
session's intervention lines past the job's start offset, returns right away
when no follow-up was started, and waits for the outcome when one was."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from scenarios import run_suite


class AwaitInterventionsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = Path(self.dir.name) / "log.jsonl"
        self.log.write_text(json.dumps({"type": "intervention", "sessionKey": "s1", "action": "verify", "scheduled": True}) + "\n",
                            encoding="utf-8")  # before the job: must be ignored
        self.saved = run_suite.PLUGIN_LOG
        run_suite.PLUGIN_LOG = self.log
        self.offset = run_suite._log_size()

    def tearDown(self):
        run_suite.PLUGIN_LOG = self.saved
        self.dir.cleanup()

    def append(self, **entry):
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "intervention", **entry}) + "\n")

    def test_no_follow_up_returns_the_decision(self):
        self.append(sessionKey="s2", action="verify", scheduled=True)  # another session
        self.append(sessionKey="s1", action="none", scheduled=False, rule="skip-verify-wastes", probability=0.9)
        iv = run_suite._await_intervention("s1", self.offset, timeout=5, decision_wait=2)
        self.assertEqual((iv["action"], iv["scheduled"], iv["rule"]), ("none", False, "skip-verify-wastes"))

    def test_waits_for_the_follow_up_outcome(self):
        self.append(sessionKey="s1", action="verify", scheduled=True, probability=0.9)
        threading.Timer(1.0, lambda: self.append(sessionKey="s1", action="outcome", success=True, toolCalls=2)).start()
        iv = run_suite._await_intervention("s1", self.offset, timeout=10, decision_wait=2)
        self.assertEqual((iv["action"], iv["outcome_success"], iv["outcome_tool_calls"]), ("verify", True, 2))
        self.assertGreater(iv["followup_seconds"], 0)

    def test_no_decision_at_all(self):
        t = time.time()
        iv = run_suite._await_intervention("s1", self.offset, timeout=5, decision_wait=1)
        self.assertIsNone(iv["action"])
        self.assertLess(time.time() - t, 5)


if __name__ == "__main__":
    unittest.main()
