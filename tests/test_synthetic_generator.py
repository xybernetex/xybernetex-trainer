import tempfile
import unittest
from pathlib import Path

import numpy as np

from chassis.action_space import ActionSpace
from chassis.state_vectorizer import STATE_DIM
from data import synthetic_generator as gen


class TestSyntheticGenerator(unittest.TestCase):
    def test_output_shapes_and_dtypes(self):
        states, actions = gen.generate_episodes(num_episodes=20)
        self.assertEqual(states.shape[1], STATE_DIM)
        self.assertEqual(states.shape[0], actions.shape[0])
        self.assertEqual(states.dtype, np.float32)
        self.assertEqual(actions.dtype, np.int64)

    def test_same_seed_is_reproducible(self):
        states_a, actions_a = gen.generate_episodes(num_episodes=20, seed=7)
        states_b, actions_b = gen.generate_episodes(num_episodes=20, seed=7)
        np.testing.assert_array_equal(states_a, states_b)
        np.testing.assert_array_equal(actions_a, actions_b)

    def test_all_actions_appear_with_enough_episodes(self):
        _, actions = gen.generate_episodes(num_episodes=300)
        seen = {ActionSpace.decode(a) for a in np.unique(actions)}
        for expected in ("CONTINUE", "REPLAN", "BLOCK_ACTION", "REQUEST_USER", "TERMINATE"):
            self.assertIn(expected, seen, f"{expected} never appeared in 300 synthetic episodes")

    def test_actions_are_valid_indices(self):
        _, actions = gen.generate_episodes(num_episodes=20)
        self.assertTrue(np.all(actions >= 0))
        self.assertTrue(np.all(actions < ActionSpace.N))

    def test_generate_and_save_writes_loadable_npz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "rollouts.npz")
            gen.generate_and_save(num_episodes=5, output_path=out_path)
            # np.load() keeps the zip file handle open until closed - on
            # Windows that blocks the tempdir cleanup below unless this is
            # a context manager (or .close() is called explicitly).
            with np.load(out_path) as data:
                self.assertEqual(set(data.files), {"states", "actions"})
                self.assertEqual(data["states"].shape[1], STATE_DIM)
                self.assertEqual(data["states"].shape[0], data["actions"].shape[0])


if __name__ == "__main__":
    unittest.main()
