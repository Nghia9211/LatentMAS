"""CPU regressions for controller package layout and relocated source locks."""
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('job_gate', 'matched', 'metrics', 'normalized', 'ppo',
           'profile', 'sampler', 'tracking', 'training', 'uniform_explore', 'warmstart')


class ControllerPackageTests(unittest.TestCase):
    def test_core_modules_live_only_in_package(self):
        for name in MODULES:
            with self.subTest(module=name):
                module = importlib.import_module('controller.' + name)
                self.assertEqual(Path(module.__file__).resolve(), ROOT / 'controller' / (name + '.py'))
                self.assertFalse((ROOT / ('controller_' + name + '.py')).exists())

    def test_gates_do_not_import_torch_or_gpu_sdk(self):
        script = (
            'import json, sys; import controller.job_gate; '
            'print(json.dumps(sorted(set(sys.modules) & {"torch", "vllm", "wandb"})))'
        )
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                                check=True, capture_output=True, text=True, timeout=30)
        self.assertEqual(json.loads(result.stdout), [])

    def test_worker_extension_resolves_from_both_current_configs(self):
        from controller.profile import ProfileWorkerInspector
        from scripts.profile_controller import load_config
        for name in ('controller_ppo_matched.yaml', 'profile_medqa_pilot.yaml'):
            cfg = yaml.safe_load((ROOT / 'configs' / name).read_text())
            dotted = cfg['worker_extension_cls']
            module, attr = dotted.rsplit('.', 1)
            self.assertIs(getattr(importlib.import_module(module), attr), ProfileWorkerInspector)
            self.assertEqual(dotted, 'controller.profile.ProfileWorkerInspector')
        load_config(ROOT / 'configs/profile_medqa_pilot.yaml')

    def test_new_source_snapshots_lock_package_initializer_and_existing_files(self):
        from scripts import profile_controller
        from scripts import run_matched_controller_ppo, run_pooled_controller_ppo
        from scripts import profile_controller_pool, train_normalized_controller
        from scripts import train_controller_from_profile
        base = run_pooled_controller_ppo.SOURCES
        self.assertIn('controller/training.py', run_matched_controller_ppo.SOURCES)
        groups = [profile_controller.SOURCES,
                  run_matched_controller_ppo.SOURCES, base, profile_controller_pool.SOURCES,
                  base + train_controller_from_profile.EXTRA_SOURCES,
                  base + train_normalized_controller.EXTRA_SOURCES]
        for sources in groups:
            self.assertIn('controller/__init__.py', sources)
            for name in sources:
                with self.subTest(path=name):
                    self.assertTrue((ROOT / name).is_file(), name)

    def test_retired_benchmark_and_pilot_entrypoints_have_no_dependents(self):
        retired = ('benchmark_controller_ppo', 'benchmark_controller_warmstart',
                   'benchmark_ppo_fresh_train_iteration', 'evaluate_controller_test16',
                   'summarize_controller_profile',
                   'audit_medqa_repeat_correctness',
                   'evaluate_profile_trained_test16',
                   'train_expanded_duration_ablation',
                   'train_expanded_lr_ablation',
                   'train_simple_state_ppo',
                   'train_uniform_explore_controller',
                   'verify_normalized_controller')
        for name in retired:
            self.assertFalse((ROOT / 'scripts' / (name + '.py')).exists())
        for directory in ('controller', 'scripts', 'tests'):
            for path in (ROOT / directory).glob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom):
                        self.assertNotIn(node.module, {'scripts.' + n for n in retired}, str(path))
                        if node.module == 'scripts':
                            for alias in node.names:
                                self.assertNotIn(alias.name, retired, str(path))
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertNotIn(alias.name, {'scripts.' + n for n in retired}, str(path))

    def test_retired_fresh_benchmark_cli_is_explicitly_rejected(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_ppo_with_wandb.py'),
                                 '--runner', 'fresh', '--help'], cwd=ROOT, capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)

    def test_active_python_has_no_legacy_root_imports(self):
        legacy = {'controller_' + name for name in MODULES}
        paths = list(ROOT.glob('*.py'))
        for directory in ('controller', 'scripts', 'tests'):
            paths.extend((ROOT / directory).glob('*.py'))
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, legacy, str(path))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name, legacy, str(path))


if __name__ == '__main__':
    unittest.main()
