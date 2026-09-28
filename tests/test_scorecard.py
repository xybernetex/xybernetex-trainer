import unittest

from scenarios.scorecard import compare, mcnemar_p, summarize


def row(key, success, tokens=100, seconds=10.0, category="hard", harm=None, flagged=False, blocked=False):
    return {"key": key, "category": category, "success": success, "tokens": tokens, "seconds": seconds,
            "harm": harm, "flagged_unrequested": flagged, "blocked": blocked}


class TestScorecard(unittest.TestCase):
    def test_mcnemar_exact_values(self):
        self.assertEqual(mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_p(0, 5), 2 / 32)       # all 5 discordant pairs one way
        self.assertAlmostEqual(mcnemar_p(1, 4), 2 * 6 / 32)   # P(X<=1 | n=5) = 6/32, two-sided
        self.assertEqual(mcnemar_p(3, 3), 1.0)

    def test_failures_carry_their_tokens_and_time(self):
        s = summarize([row("a", True, 100, 10), row("b", False, 300, 50), row("c", None, 999, 999)])
        self.assertEqual((s["scored"], s["successes"]), (2, 1))
        self.assertEqual(s["tokens_per_success"], 400)   # the failed run's tokens count, the unscored one's don't
        self.assertEqual(s["seconds_per_success"], 60)

    def test_safety_counts(self):
        s = summarize([row("h1", None, category="injection", harm=True, flagged=True),
                       row("h2", True, category="injection", harm=True),
                       row("ok", True, blocked=True), row("ok2", True)])
        self.assertEqual((s["harmful_runs"], s["harmful_flagged"]), (2, 1))
        self.assertEqual((s["false_block_runs"], s["clean_successes"]), (1, 2))

    def test_paired_comparison(self):
        a = [row(k, k < 6) for k in range(10)]
        b = [row(k, k < 9, tokens=120) for k in range(10)]
        c = compare(a, b, rounds=200)
        self.assertEqual((c["pairs"], c["only_a"], c["only_b"]), (10, 0, 3))
        self.assertAlmostEqual(c["tokens_ratio"], (1200 / 9) / (1000 / 6))
        self.assertLessEqual(c["tokens_ci"][0], c["tokens_ratio"])
        self.assertGreaterEqual(c["tokens_ci"][1], c["tokens_ratio"])

    def test_unpaired_jobs_are_ignored(self):
        c = compare([row(1, True), row(2, True)], [row(2, False), row(3, True)], rounds=50)
        self.assertEqual((c["pairs"], c["only_a"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
