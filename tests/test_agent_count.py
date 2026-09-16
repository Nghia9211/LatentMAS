import unittest
from methods import default_agents


class AgentCountTests(unittest.TestCase):
    def test_two_agents(self):
        self.assertEqual([a.role for a in default_agents(2)], ["planner", "judger"])

    def test_original_four_agents(self):
        self.assertEqual([a.role for a in default_agents(4)],
                         ["planner", "critic", "refiner", "judger"])

    def test_invalid_count(self):
        with self.assertRaises(ValueError):
            default_agents(3)


if __name__ == "__main__":
    unittest.main()
