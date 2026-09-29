"""run_suite --counterfactual, with the agent CLI, the grader and the session
store stubbed: every branch starts from the same snapshot, fresh branches get
their own folder, session and restated task, the strong branch on the strong
model is an alias, dead runs get retries, and timed-out runs get nothing."""
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scenarios import run_suite as rs


class HarnessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        self.calls, self.lock = [], threading.Lock()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, status="done", model="gemma4"):
        job = {"session_key": "agent:scenarios:scn-b-x-gemma4-r1", "folder": "runs/b/x/gemma4-r1", "model": model}
        folder = self.ws / job["folder"]
        folder.mkdir(parents=True)
        (folder / "out.txt").write_text("first answer", encoding="utf-8")

        def agent_turn(bjob, oc, timeout, message):
            with self.lock:
                self.calls.append((bjob["session_key"], bjob["model"], bjob["folder"], message))
            # Every follow-up edits its own folder; none may see another's edit.
            f = self.ws / bjob["folder"]
            seen = (f / "out.txt").read_text(encoding="utf-8")
            (f / "out.txt").write_text(seen + " + " + bjob["session_key"].rsplit("-", 1)[-1], encoding="utf-8")
            return "done", 0, '{"result": {"payloads": [{"text": "ok"}]}}', ""

        def evaluate(scenario, folder, reply):
            text = (Path(folder) / "out.txt").read_text(encoding="utf-8")
            return SimpleNamespace(task=text.count("+") == 1 and "same" not in text, detail=text)

        with mock.patch.object(rs, "WORKSPACE", self.ws), mock.patch.object(rs, "_agent_turn", agent_turn), \
                mock.patch("scenarios.outcomes.evaluate", evaluate), mock.patch.object(
                    rs, "_transcript", lambda k, s: [{"role": "assistant", "usage": {"totalTokens": 111}}]):
            return rs._counterfactual(job, {"prompt": "do x"}, [], 60, {"status": status, "reply": "hi"}, 0.0), folder

    def test_finished_run_every_branch_from_the_same_snapshot(self):
        cf, folder = self._run()
        self.assertEqual(cf["kind"], "finished")
        self.assertFalse(cf["before"]["task"])  # the snapshot: first answer, no follow-up
        self.assertEqual(set(cf["branches"]), {"verify-same", "verify-fresh", "test-fresh", "verify-strong"})
        self.assertEqual((folder.with_name(folder.name + ".before") / "out.txt").read_text(encoding="utf-8"), "first answer")
        for name in ("verify-fresh", "test-fresh", "verify-strong"):
            text = (folder.with_name(f"{folder.name}.{name}") / "out.txt").read_text(encoding="utf-8")
            self.assertEqual(text.count("+"), 1, (name, text))  # started from the snapshot, saw no other branch
        by_key = {c[0]: c for c in self.calls}
        self.assertEqual(by_key["agent:scenarios:scn-b-x-gemma4-r1"][3], rs.FOLLOW_UPS["verify"])  # same session
        strong = by_key["agent:scenarios:scn-b-x-gemma4-r1-verify-strong"]
        self.assertEqual(strong[1], rs.STRONG_MODEL)
        self.assertIn("Task: do x", strong[3])                       # fresh sessions get the task restated
        self.assertIn("write a small test script", by_key["agent:scenarios:scn-b-x-gemma4-r1-test-fresh"][3].lower())
        self.assertTrue(cf["branches"]["verify-fresh"]["task"])
        # The same-session branch works in the run's own folder (as the plugin's follow-up does).
        self.assertEqual((folder / "out.txt").read_text(encoding="utf-8"), "first answer + r1")
        self.assertEqual(cf["branches"]["verify-same"]["model"], "gemma4")
        # Every branch is charged its own follow-up's tokens - the same-session one included
        # (it once came out 0: the since-start transcript has no second user message to split at).
        self.assertEqual({n: b["tokens"] for n, b in cf["branches"].items()},
                         {"verify-same": 111, "verify-fresh": 111, "test-fresh": 111, "verify-strong": 111})

    def test_the_strong_branch_on_the_strong_model_reuses_the_fresh_one(self):
        cf, _ = self._run(model=rs.STRONG_MODEL)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(cf["branches"]["verify-strong"]["alias"], "verify-fresh")

    def test_dead_runs_get_retries_and_timed_out_runs_get_nothing(self):
        cf, _ = self._run(status="failed")
        self.assertEqual(cf["kind"], "died")
        self.assertEqual(set(cf["branches"]), {"retry-same", "retry-fresh", "retry-strong"})
        self.assertFalse(cf["before"]["task"])  # a dead run gave no answer
        self.calls.clear()
        self.tmp.cleanup()
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        cf, _ = self._run(status="timeout")
        self.assertEqual((cf["kind"], cf["branches"], self.calls), (None, {}, []))


if __name__ == "__main__":
    unittest.main()


class FreshAttemptTest(unittest.TestCase):
    """A job re-run after a crash must not continue its earlier attempt's session."""

    def test_rerun_takes_first_unused_attempt_key(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "sessions.sqlite"
            conn = sqlite3.connect(db)
            conn.execute("create table session_windows (session_id text, session_key text)")
            conn.executemany("insert into session_windows values (?, ?)", [
                ("s1", "agent:scenarios:scn-b-x-m-r1"), ("s2", "agent:scenarios:scn-b-x-m-r1-a2-verify-fresh"),
                ("s3", "agent:scenarios:scn-b-y-m-r1-retry-fresh")])
            conn.commit()
            conn.close()
            with mock.patch("scenarios.outcomes.SESSION_DB", db):
                fresh = rs._fresh_attempt({"session_key": "agent:scenarios:scn-b-z-m-r1"})
                crashed = rs._fresh_attempt({"session_key": "agent:scenarios:scn-b-x-m-r1"})
                branch_only = rs._fresh_attempt({"session_key": "agent:scenarios:scn-b-y-m-r1"})
        self.assertEqual(fresh, {"session_key": "agent:scenarios:scn-b-z-m-r1"})
        self.assertEqual(crashed, {"session_key": "agent:scenarios:scn-b-x-m-r1-a3",
                                   "base_key": "agent:scenarios:scn-b-x-m-r1"})
        self.assertEqual(branch_only["session_key"], "agent:scenarios:scn-b-y-m-r1-a2")
