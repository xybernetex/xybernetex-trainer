import unittest

import torch

from chassis.action_space import ActionSpace
from chassis.actor_critic import ActorCritic, count_parameters, load_policy_bytes, save_policy_bytes
from chassis.state_vectorizer import TRAINING_STATE_DIM


class TestActorCriticShapes(unittest.TestCase):
    def test_defaults_match_chassis_constants(self):
        model = ActorCritic()
        self.assertEqual(model.state_dim, TRAINING_STATE_DIM)
        self.assertEqual(model.action_dim, ActionSpace.N)

    def test_forward_single_input_shapes(self):
        model = ActorCritic()
        x = torch.randn(TRAINING_STATE_DIM)
        logits, value = model(x)
        self.assertEqual(logits.shape, (ActionSpace.N,))
        self.assertEqual(value.shape, (1,))

    def test_forward_batched_input_shapes(self):
        model = ActorCritic()
        x = torch.randn(4, TRAINING_STATE_DIM)
        logits, value = model(x)
        self.assertEqual(logits.shape, (4, ActionSpace.N))
        self.assertEqual(value.shape, (4, 1))

    def test_save_load_roundtrip_preserves_output(self):
        model = ActorCritic()
        model.eval()
        x = torch.randn(TRAINING_STATE_DIM)
        with torch.no_grad():
            logits_before, value_before = model(x)

        reloaded = load_policy_bytes(save_policy_bytes(model))
        with torch.no_grad():
            logits_after, value_after = reloaded(x)

        torch.testing.assert_close(logits_before, logits_after)
        torch.testing.assert_close(value_before, value_after)

    def test_parameter_count_is_small_enough_for_edge_serving(self):
        model = ActorCritic()
        n_params = count_parameters(model)
        # Sanity bound, not an exact pin - just confirms this stays in the
        # "small enough for a pure-JS Worker forward pass" ballpark rather
        # than silently growing into something that needs a real runtime.
        self.assertLess(n_params, 500_000)
        self.assertGreater(n_params, 1_000)


if __name__ == "__main__":
    unittest.main()
