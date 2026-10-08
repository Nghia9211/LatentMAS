"""Seed plumbing tests with mocked vLLM; no model downloads or GPU workloads."""
import ast
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from methods import default_agents
from utils import set_seed


ROOT = Path(__file__).parents[1]


def load_method(filename, class_name, method_name, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == method_name)
    future = ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    exec(compile(module, str(ROOT / filename), "exec"), namespace)
    return namespace[method_name]


class SeedTests(unittest.TestCase):
    def test_global_rng_repeatability(self):
        states = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            with patch.dict(os.environ), patch.object(torch.cuda, "manual_seed_all") as cuda_seed:
                set_seed()
                first = random.random(), np.random.rand(), torch.rand(3)
                set_seed(42)
                second = random.random(), np.random.rand(), torch.rand(3)
                self.assertEqual(first[:2], second[:2])
                self.assertTrue(torch.equal(first[2], second[2]))
                self.assertEqual(os.environ["PYTHONHASHSEED"], "42")
                cuda_seed.assert_called_with(42)
        finally:
            random.setstate(states[0])
            np.random.set_state(states[1])
            torch.set_rng_state(states[2])

    def test_engine_and_text_sampling_seed(self):
        for method_name, prefix in [("latent_mas", True), ("latent_mas", False), ("text_mas", True),
                                    ("baseline", False)]:
            for seed in (None, 42, 123):
                with self.subTest(method=method_name, prefix=prefix, seed=seed):
                    engine_factory = MagicMock()
                    ns = dict(_HAS_VLLM=True, LLM=engine_factory,
                              AutoTokenizer=MagicMock(), _ensure_pad_token=lambda _: None,
                              AutoModelForCausalLM=MagicMock(), torch=torch,
                              SamplingParams=SimpleNamespace)
                    init = load_method("models.py", "ModelWrapper", "__init__", ns)
                    generate = load_method("models.py", "ModelWrapper",
                                           "vllm_generate_text_batch", ns)
                    args = SimpleNamespace(method=method_name, enable_prefix_caching=prefix, device2='cpu')
                    if seed is not None:
                        args.seed = seed
                    model = SimpleNamespace(_ensure_latent_realign_matrix=MagicMock())
                    init(model, "fake-model", "cpu", use_vllm=True, args=args)
                    expected = 42 if seed is None else seed
                    self.assertEqual(engine_factory.call_args.kwargs["seed"], expected)
                    self.assertEqual(engine_factory.call_args.kwargs['enable_prefix_caching'], prefix)
                    if method_name == 'latent_mas':
                        self.assertTrue(engine_factory.call_args.kwargs['enable_prompt_embeds'])
                    model.vllm_engine.generate.return_value = [
                        SimpleNamespace(outputs=[SimpleNamespace(text="answer")])]
                    self.assertEqual(generate(model, ["prompt"]), ["answer"])
                    params = model.vllm_engine.generate.call_args.args[1]
                    self.assertEqual(params.seed, expected)

    def test_latent_sampling_seed(self):
        ns = dict(default_agents=default_agents, SamplingParams=SimpleNamespace)
        init = load_method("methods/latent_mas.py", "LatentMASMethod", "__init__", ns)
        for seed in (None, 42, 123):
            with self.subTest(seed=seed):
                args = SimpleNamespace(device="cpu", device2="cpu", task="medqa",
                                       max_new_tokens=128, num_agents=2)
                if seed is not None:
                    args.seed = seed
                method = SimpleNamespace()
                init(method, object(), args=args)
                self.assertEqual(method.sampling_params.seed, 42 if seed is None else seed)


if __name__ == "__main__":
    unittest.main()
