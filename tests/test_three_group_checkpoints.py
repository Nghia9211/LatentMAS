"""Checkpoint dispatch must distinguish same-sized L2 and cosine states."""
import tempfile
from pathlib import Path
import unittest

import torch

from controller import simple_state, cosine_state
from controller.checkpoints import load_controller_checkpoint


class ThreeGroupCheckpointTests(unittest.TestCase):
    def test_dispatch_roundtrip_and_schema_guard(self):
        for module, name in [(simple_state, "Simple"), (cosine_state, "Cosine")]:
            with self.subTest(schema=module.SCHEMA):
                torch.manual_seed(42)
                model = getattr(module, name + "PPOController")()
                x = torch.randn(3, 7, 2562)
                model.normalizer.fit(x, source_split="train")
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "checkpoint.pt"
                    torch.save({"model": model.state_dict(), "iteration": 0}, path)
                    loaded = load_controller_checkpoint(path, expected_schema=module.SCHEMA)
                    self.assertIsInstance(loaded, type(model))
                    for actual, expected in zip(loaded(x), model(x)):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    wrong = cosine_state.SCHEMA if module is simple_state else simple_state.SCHEMA
                    with self.assertRaises(ValueError):
                        load_controller_checkpoint(path, expected_schema=wrong)
                    saved = torch.load(path, weights_only=True)
                    saved["model"]["normalizer.schema_version"].fill_(999)
                    torch.save(saved, path)
                    with self.assertRaises(ValueError):
                        load_controller_checkpoint(path)

    def test_runtime_memory_rechecks_after_tracking_without_loading_models(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from controller.online_three_group import _wait_for_model_memory

        gib = 1024**3
        # First memory sample is too small, second is sufficient.
        replies = [(10*gib, 80*gib), (10*gib, 80*gib),
                   (30*gib, 80*gib), (30*gib, 80*gib)]
        with patch("torch.cuda.mem_get_info", side_effect=replies), \
             patch("controller.online_three_group.time.sleep") as sleep, \
             patch("controller.online_three_group.event") as event:
            _wait_for_model_memory(SimpleNamespace(device2="cuda:1", gpu_memory_utilization=.18), Path("unused"))
            self.assertEqual(sleep.call_count, 1)
            self.assertEqual(event.call_args.kwargs["stage"], "model_memory_admitted")


if __name__ == "__main__":
    unittest.main()
