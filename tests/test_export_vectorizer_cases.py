import json
import unittest

import numpy as np

from chassis.state_vectorizer import STATE_DIM, vectorize
from data.trajectory_schema import ToolCallRecord, TrajectorySnapshot
from export.export_vectorizer_cases import build_cases


class ExportVectorizerCasesTest(unittest.TestCase):
    def test_cases_round_trip_through_json_to_the_same_features(self):
        payload = json.loads(json.dumps(build_cases()))
        self.assertEqual(payload["state_dim"], STATE_DIM)
        names = [case["name"] for case in payload["cases"]]
        self.assertEqual(len(names), len(set(names)))
        for case in payload["cases"]:
            raw = dict(case["snapshot"])
            raw["tool_history"] = [ToolCallRecord(**call) for call in raw["tool_history"]]
            # The JSON form is what the Worker receives, so it must rebuild a
            # snapshot that vectorizes to exactly the recorded features.
            with np.errstate(over="ignore"):
                features = vectorize(TrajectorySnapshot(**raw)).tolist()
            self.assertEqual(features, case["features"], case["name"])

    def test_covers_every_synthetic_category(self):
        names = [case["name"] for case in build_cases()["cases"]]
        for category in ("normal", "stuck_loop", "error_recovery", "risky_tool", "budget_exhaustion"):
            self.assertTrue(any(name.startswith(f"synthetic {category} ") for name in names), category)


if __name__ == "__main__":
    unittest.main()
