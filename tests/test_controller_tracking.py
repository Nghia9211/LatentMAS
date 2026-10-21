import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from controller.tracking import TrainingTracker, require_wandb
from controller.ppo import PPOController, sample_episodes
from controller.training import logged_ppo_update
from scripts.run_matched_controller_ppo import ppo_update
from scripts.run_ppo_with_wandb import PPOLoggingSession

CONFIG = dict(optimization_epochs=4, minibatch_questions=8, clip=.2,
              value_weight=.5, entropy_weight=.01, tie_weight=.1, grad_clip=1)


class FakeRun:
    def __init__(self):
        self.metrics, self.rows, self.closed = [], [], []
        self.fail = False
    def define_metric(self, name, **kwargs):
        self.metrics.append((name, kwargs))
    def log(self, row):
        if self.fail:
            raise RuntimeError('offline test failure')
        self.rows.append(row)
    def finish(self, **kwargs):
        self.closed.append(kwargs)


class FakeSDK:
    def __init__(self):
        self.run = FakeRun()
        self.kwargs = None
    def Settings(self, **kwargs):
        return kwargs
    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run


def example():
    torch.manual_seed(42)
    model = PPOController(9, 16)
    x = torch.randn(32, 7, 9)
    episode = sample_episodes(model, x, torch.randint(0, 2, (32, 8)))
    teacher = torch.randn_like(x)
    ties = torch.ones(32, 7, dtype=torch.bool)
    return model, x, episode, teacher, ties


class TrackingTests(unittest.TestCase):
    def test_disabled_does_not_import_sdk_and_saves_local_scalars(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('controller.tracking.importlib.import_module', side_effect=AssertionError('SDK imported')):
                tracker = TrainingTracker(directory)
                tracker.log({'optimizer_step': 1, 'loss/total': -.125})
                tracker.finish()
            row = json.loads((Path(directory) / 'training_metrics.jsonl').read_text())
            self.assertEqual(row['loss/total'], -.125)
            self.assertIsNone(tracker.run)

    def test_sdk_axes_scalar_only_and_privacy_options(self):
        sdk = FakeSDK()
        with tempfile.TemporaryDirectory() as directory:
            tracker = TrainingTracker(directory, mode='offline', sdk=sdk,
                config={'seed': 42, 'question': 'secret', 'gold': 'secret',
                        'data_dir': '/private/data', 'api_key': 'secret'})
            self.assertEqual(sdk.kwargs['config'], {'seed': 42})
            self.assertEqual(sdk.kwargs['mode'], 'offline')
            self.assertEqual(sdk.kwargs['settings']['console'], 'off')
            self.assertTrue(sdk.kwargs['settings']['disable_git'])
            self.assertFalse(sdk.kwargs['save_code'])
            self.assertIn(('loss/*', {'step_metric': 'optimizer_step'}), sdk.run.metrics)
            self.assertIn(('train/*', {'step_metric': 'iteration'}), sdk.run.metrics)
            tracker.log({'optimizer_step': 3, 'loss/total': .1})
            with self.assertRaises(TypeError):
                tracker.log({'question': 'do not upload'})
            tracker.finish(0)
            self.assertEqual(sdk.run.closed, [{'exit_code': 0}])
            self.assertEqual(len(sdk.run.rows), 1)

    def test_remote_failure_retains_local_logging(self):
        sdk = FakeSDK()
        sdk.run.fail = True
        with tempfile.TemporaryDirectory() as directory:
            tracker = TrainingTracker(directory, mode='offline', sdk=sdk)
            with self.assertWarns(RuntimeWarning):
                tracker.log({'loss/total': 1.})
            tracker.log({'loss/total': 2.})
            self.assertEqual(len((Path(directory) / 'training_metrics.jsonl').read_text().splitlines()), 2)
            tracker.finish()

    def test_online_resume_identity_and_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            first = FakeSDK()
            a = TrainingTracker(directory, mode='online', sdk=first)
            a.log({'optimizer_step': 1})
            a.finish()
            second = FakeSDK()
            b = TrainingTracker(directory, mode='online', sdk=second)
            self.assertEqual(first.kwargs['id'], second.kwargs['id'])
            self.assertEqual(second.kwargs['resume'], 'allow')
            b.log({'optimizer_step': 2})
            self.assertEqual(second.run.rows[0]['tracking/attempt'], 2)
            b.finish()
            with self.assertRaises(ValueError):
                TrainingTracker(directory, mode='online', project='different', sdk=FakeSDK())

    def test_missing_optional_dependency_explains_install(self):
        with patch('controller.tracking.importlib.import_module', side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, 'requirements-tracking.txt'):
                require_wandb('online')
        with self.assertRaises(ValueError):
            require_wandb('invalid')

    def test_instrumented_update_exactly_matches_legacy_math_and_rng(self):
        model, x, episode, teacher, ties = example()
        other = copy.deepcopy(model)
        opt_a = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=0)
        opt_b = torch.optim.AdamW(other.parameters(), lr=.0003, weight_decay=0)
        rng = torch.get_rng_state()
        expected = ppo_update(model, opt_a, x, episode, teacher, ties, CONFIG)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        rows = []
        actual = logged_ppo_update(other, opt_b, x, episode, teacher, ties, CONFIG,
                                   lambda step, metrics: rows.append((step, metrics)))
        self.assertEqual(expected, actual)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items()))
        self.assertEqual([i for i, _ in rows], list(range(1, 17)))
        for _, row in rows:
            self.assertAlmostEqual(row['loss/total'], row['loss/policy'] +
                .5 * row['loss/value'] + row['loss/entropy_term'] + .1 * row['loss/tie'], places=6)

    def test_fresh_session_logs_absolute_update_steps_and_committed_mean(self):
        model, x, episode, teacher, ties = example()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps(dict(parent_config={'seed': 42}, parent_iteration=7)))
            options = SimpleNamespace(wandb_mode='disabled', wandb_project='test',
                                      wandb_entity=None, wandb_name=None)
            session = PPOLoggingSession(root, options, 'fresh')
            mean, updates = session.update(model, optimizer, x, episode, teacher, ties, CONFIG)
            session.event(lambda *args, **kwargs: None, root, stage='one_iteration_complete',
                total_controller_iterations=8, correct=22, mean_selected_K=13.75,
                seconds_including_live_A_B=286.)
            session.close(0)
            rows = list(map(json.loads, (root / 'training_metrics.jsonl').read_text().splitlines()))
            self.assertEqual(len(rows), 17)
            self.assertEqual([r['optimizer_step'] for r in rows[:-1]], list(range(113, 129)))
            self.assertTrue(all(r['iteration'] == 8 for r in rows))
            self.assertEqual(rows[-1]['iteration/loss_total'], mean)
            self.assertEqual(rows[-1]['train/accuracy'], 22 / 32)
            self.assertEqual(updates, 16)


if __name__ == '__main__':
    unittest.main()
