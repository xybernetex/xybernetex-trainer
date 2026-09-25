import unittest


@unittest.skip("chassis/actor_critic.py not implemented yet - waiting on state_vectorizer.py to fix STATE_DIM")
class TestActorCriticShapes(unittest.TestCase):
    def test_placeholder(self):
        pass


if __name__ == "__main__":
    unittest.main()
