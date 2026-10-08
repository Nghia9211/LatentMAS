import unittest
import argparse
import ast
from pathlib import Path
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

    def test_textmas_constructor_uses_requested_count(self):
        # Test the actual constructor without importing model/GPU dependencies.
        path = Path(__file__).parents[1] / "methods" / "text_mas.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "TextMASMethod")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == "__init__")
        ns = dict(default_agents=default_agents, ModelWrapper=object, argparse=argparse)
        exec(compile(ast.Module(body=[init], type_ignores=[]), str(path), "exec"), ns)
        for count, expected in [(None, ["planner", "judger"]),
                                (2, ["planner", "judger"]),
                                (4, ["planner", "critic", "refiner", "judger"])]:
            with self.subTest(count=count):
                args = argparse.Namespace(task="medqa")
                if count is not None:
                    args.num_agents = count
                instance = argparse.Namespace()
                ns["__init__"](instance, object(), args=args)
                self.assertEqual([a.role for a in instance.agents], expected)


if __name__ == "__main__":
    unittest.main()
