"""learn.fleet: customers' outcome episodes -> bandit records. The proxy
reward, logged propensities, the per-customer cap, and - on a simulated fleet
with known truth - that the learner verifies only where verifying pays."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from learn import bandit, fleet


def row(customer="acme", model="m", applied="none", p=1.0, run_success=1, user="new", followup_success=0,
        retriable=0, tokens=1000, followup_tokens=-1, tool_calls=3, sample_interval=1):
    return {"customer": customer, "model": model, "decided": applied, "applied": applied, "rule": "r", "user": user,
            "mode": "act", "probability": p, "run_success": run_success, "retriable": retriable, "tool_calls": tool_calls,
            "tokens": tokens, "followup_ran": int(applied != "none"), "followup_success": followup_success,
            "followup_tokens": followup_tokens, "sample_interval": sample_interval}


class FleetRecordsTest(unittest.TestCase):
    def test_proxy_success_is_work_standing_without_a_correction(self):
        self.assertTrue(fleet.proxy_success(row(user="thanks")))
        self.assertTrue(fleet.proxy_success(row(user="none")))
        self.assertFalse(fleet.proxy_success(row(user="correction")))
        self.assertFalse(fleet.proxy_success(row(user="repeat")))
        self.assertFalse(fleet.proxy_success(row(run_success=0, applied="none")))
        self.assertTrue(fleet.proxy_success(row(run_success=0, applied="retry", followup_success=1)))
        self.assertFalse(fleet.proxy_success(row(run_success=0, applied="retry", followup_success=1, user="correction")))

    def test_records_use_the_logged_propensity_and_skip_what_cant_fit(self):
        recs = fleet.to_records([
            row(applied="verify", p=0.9, followup_tokens=500),
            row(applied="none", p=0.1),
            row(applied="retry", p=0.9, run_success=0, retriable=1),
            row(applied="verify", p=0.9, run_success=0),     # verify on a dead run: not an available action
            row(applied="none", p=0),                        # no probability: can't be reweighted
            row(applied="none", p=1.0, sample_interval=3),   # a sampled row stands for three
        ])
        self.assertEqual(len(recs), 6)
        self.assertEqual(recs[0].behavior, {"verify": 0.9, "none": approx(0.1)})
        self.assertEqual(recs[0].tokens, 1500)  # both turns; -1 (unknown) counts as 0
        self.assertEqual(recs[1].behavior["none"], 0.1)
        self.assertEqual(recs[2].actions, ("none", "retry"))
        self.assertEqual({r.scenario for r in recs}, {"acme"})

    def test_no_customer_exceeds_the_cap(self):
        recs = fleet.to_records([row(customer="big")] * 80 + [row(customer=c) for c in ("a", "b", "c", "d") for _ in range(5)])
        kept, dropped = fleet.cap_customers(recs, cap=0.25)
        share = sum(r.scenario == "big" for r in kept) / len(kept)
        self.assertLessEqual(share, 0.25)
        self.assertGreater(dropped["big"], 0)
        self.assertEqual(sum(r.scenario != "big" for r in kept), 20)  # small customers untouched

    def test_few_customers_are_split_into_buckets_for_cross_fitting(self):
        recs = fleet.to_records([row(customer=c) for c in ("a", "b") for _ in range(10)])
        fleet.assign_folds(recs)
        self.assertEqual(len({r.scenario for r in recs}), 10)
        many = fleet.to_records([row(customer=f"c{i}") for i in range(6)])
        fleet.assign_folds(many)
        self.assertEqual({r.scenario for r in many}, {f"c{i}" for i in range(6)})


def approx(x):
    class Approx(float):
        def __eq__(self, other):
            return abs(other - x) < 1e-9
        __hash__ = float.__hash__
    return Approx(x)


def simulate_fleet(n_customers=6, per_customer=120, seed=3) -> list[dict]:
    """Verify lifts proxy success on model 'helps' (+30 points), not on 'wastes';
    the logged policy verified half the finished runs."""
    rng = random.Random(seed)
    rows = []
    for c in range(n_customers):
        for _ in range(per_customer):
            model = rng.choice(["helps", "wastes"])
            verify = rng.random() < 0.5
            ok = rng.random() < 0.55 + (0.3 if verify and model == "helps" else 0)
            rows.append(row(customer=f"c{c}", model=model, applied="verify" if verify else "none", p=0.5,
                            user="thanks" if ok else "correction", tokens=100_000,
                            followup_tokens=60_000 if verify else -1))
    return rows


class FetchTest(unittest.TestCase):
    def test_the_query_reads_the_outcomes_dataset_and_skips_smoke_tests(self):
        sent = {}

        class Res:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"data": [row()]}).encode()

        def opener(req):
            sent.update(url=req.full_url, body=req.data.decode(), auth=req.headers["Authorization"])
            return Res()

        with mock.patch.dict("os.environ", {"CF_ACCOUNT_ID": "acct", "CF_ANALYTICS_TOKEN": "tok"}):
            rows = fleet.fetch_rows(400, opener=opener)
        self.assertEqual(rows, [row()])
        self.assertTrue(sent["url"].endswith("/accounts/acct/analytics_engine/sql"))
        self.assertEqual(sent["auth"], "Bearer tok")
        self.assertIn("FROM xybernetex_outcomes", sent["body"])
        self.assertIn("blob2 != 'smoke-test'", sent["body"])
        self.assertIn("INTERVAL '90' DAY", sent["body"])  # clamped to the dataset's retention
        with mock.patch.dict("os.environ", {}, clear=True), self.assertRaises(SystemExit):
            fleet.fetch_rows(7, opener=opener)


class FleetLearningTest(unittest.TestCase):
    def test_the_learner_verifies_where_it_pays_on_a_simulated_fleet(self):
        recs, _ = fleet.cap_customers(fleet.to_records(simulate_fleet()), cap=0.25)
        table = bandit.export_policy(recs, lam=0.05, epsilon=0.0)["models"]
        self.assertEqual(table["helps"]["verify"], 1.0)
        self.assertEqual(table["wastes"]["verify"], 0.0)
        fleet.assign_folds(recs)
        folds, models = bandit.crossfit(recs)
        learned = bandit.evaluate(recs, bandit.learned_crossfit(recs, folds, models, lam=0.05), models, folds)
        never = bandit.evaluate(recs, bandit.never, models, folds)
        self.assertGreater(learned["dr_success"], never["dr_success"] + 0.08)

    def test_a_shared_table_needs_several_customers(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "rows.json"
            src.write_text(json.dumps(simulate_fleet(n_customers=2)), encoding="utf-8")
            out = Path(d) / "table.json"
            argv = ["fleet", "--from-json", str(src), "--rounds", "5", "--export", str(out)]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                with self.assertRaises(SystemExit) as ctx:
                    fleet.main()
            self.assertIn("too few for a shared table", str(ctx.exception))
            self.assertFalse(out.exists())
            with mock.patch.object(sys, "argv", argv + ["--allow-few-customers"]), mock.patch("builtins.print"):
                fleet.main()
            self.assertEqual(json.loads(out.read_text())["source"]["kind"], "fleet-proxy")


if __name__ == "__main__":
    unittest.main()
