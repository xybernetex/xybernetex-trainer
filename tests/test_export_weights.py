import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from chassis.actor_critic import ActorCritic
from chassis.state_vectorizer import STATE_DIM
from export.export_weights import pack_and_write, pack_weights, reference_forward


class TestExportWeights(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.model = ActorCritic()
        self.model.eval()

    def test_manifest_tensor_list_matches_state_dict_exactly(self):
        manifest, _raw_bytes = pack_weights(self.model)
        manifest_names = [t["name"] for t in manifest["tensors"]]
        self.assertEqual(manifest_names, list(self.model.state_dict().keys()))
        for t in manifest["tensors"]:
            expected_shape = list(self.model.state_dict()[t["name"]].shape)
            self.assertEqual(t["shape"], expected_shape)

    def test_raw_bytes_length_matches_total_parameter_count(self):
        manifest, raw_bytes = pack_weights(self.model)
        total_floats = sum(t["length"] for t in manifest["tensors"])
        self.assertEqual(len(raw_bytes), total_floats * 4)  # float32 = 4 bytes

    def test_manifest_dims_match_model(self):
        manifest, _raw_bytes = pack_weights(self.model)
        self.assertEqual(manifest["state_dim"], self.model.state_dim)
        self.assertEqual(manifest["action_dim"], self.model.action_dim)

    def test_reference_forward_matches_real_pytorch_model(self):
        """The core round-trip check: pack the model, reconstruct a forward
        pass from ONLY the packed bytes (no PyTorch), and confirm it agrees
        with the real model.forward() output. This is the exact numerical
        bar xybernetex-cfworker's src/policy.js has to clear too."""
        manifest, raw_bytes = pack_weights(self.model)
        rng = np.random.default_rng(0)
        state_vector = rng.standard_normal(STATE_DIM).astype(np.float32)

        with torch.no_grad():
            real_logits, real_value = self.model(torch.tensor(state_vector))

        ref_logits, ref_value = reference_forward(manifest, raw_bytes, state_vector)

        np.testing.assert_allclose(ref_logits, real_logits.numpy(), atol=1e-5)
        self.assertAlmostEqual(ref_value, real_value.item(), places=5)

    def test_reference_forward_matches_across_multiple_random_inputs(self):
        manifest, raw_bytes = pack_weights(self.model)
        rng = np.random.default_rng(1)
        for _ in range(20):
            state_vector = rng.standard_normal(STATE_DIM).astype(np.float32)
            with torch.no_grad():
                real_logits, real_value = self.model(torch.tensor(state_vector))
            ref_logits, ref_value = reference_forward(manifest, raw_bytes, state_vector)
            np.testing.assert_allclose(ref_logits, real_logits.numpy(), atol=1e-5)
            self.assertAlmostEqual(ref_value, real_value.item(), places=5)

    def test_pack_and_write_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path, weights_path = pack_and_write(self.model, tmpdir)
            self.assertTrue(Path(manifest_path).exists())
            self.assertTrue(Path(weights_path).exists())

            import json
            manifest = json.loads(Path(manifest_path).read_text())
            raw_bytes = Path(weights_path).read_bytes()

            state_vector = np.random.default_rng(2).standard_normal(STATE_DIM).astype(np.float32)
            with torch.no_grad():
                real_logits, _real_value = self.model(torch.tensor(state_vector))
            ref_logits, _ref_value = reference_forward(manifest, raw_bytes, state_vector)
            np.testing.assert_allclose(ref_logits, real_logits.numpy(), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
