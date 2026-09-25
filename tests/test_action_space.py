import unittest

from chassis.action_space import ActionSpace


class TestActionSpace(unittest.TestCase):
    def test_n_matches_primitives_length(self):
        self.assertEqual(ActionSpace.N, len(ActionSpace.PRIMITIVES))
        self.assertEqual(ActionSpace.N, 8)

    def test_encode_decode_roundtrip(self):
        for action in ActionSpace.PRIMITIVES:
            self.assertEqual(ActionSpace.decode(ActionSpace.encode(action)), action)

    def test_decode_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            ActionSpace.decode(-1)
        with self.assertRaises(ValueError):
            ActionSpace.decode(ActionSpace.N)

    def test_encode_rejects_unknown_action(self):
        with self.assertRaises(ValueError):
            ActionSpace.encode("NOT_A_REAL_ACTION")

    def test_mask_marks_only_valid_actions(self):
        mask = ActionSpace.mask(["CONTINUE", "TERMINATE"])
        self.assertEqual(mask.sum(), 2)
        self.assertTrue(mask[ActionSpace.encode("CONTINUE")])
        self.assertTrue(mask[ActionSpace.encode("TERMINATE")])
        self.assertFalse(mask[ActionSpace.encode("BLOCK_ACTION")])

    def test_sample_is_always_a_valid_index(self):
        for _ in range(50):
            index = ActionSpace.sample()
            self.assertTrue(0 <= index < ActionSpace.N)


if __name__ == "__main__":
    unittest.main()
