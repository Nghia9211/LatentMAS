import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from controller.normalized import (ObservationNormalizer, NormalizedPPOController,
    NormalizedWarmStartController, load_normalized_checkpoint, encoder_diagnostics)
from controller.ppo import PPOController, sample_episodes, ppo_loss, greedy_budgets, OnlineStopper
from controller.profile import raw_observation
from controller.warmstart import warmstart_loss
from controller.matched import CohortDecisions
from controller.training import logged_ppo_update
from scripts.train_normalized_controller import load_config


def features(n=8):
    x = torch.randn(n, 7, 17)
    x[..., :12] *= 30
    x[..., -5:-1] = x[..., -5:-1]*torch.tensor([.1, 2., .01, 5.])+torch.tensor([.9, 1., .1, 2.])
    x[..., -1] = torch.arange(1, 8)/8
    return x


def model_with_stats(x, **kwargs):
    m = NormalizedPPOController(x.shape[-1], 8, **kwargs)
    m.normalizer.fit(x, source_split='train')
    return m


class NormalizerTests(unittest.TestCase):
    def test_blockwise_normalization_phi_and_metadata(self):
        x = features()
        before = x.clone()
        m = ObservationNormalizer(17).fit(x, source_split='train')
        y = m(x)
        for i in range(3):
            torch.testing.assert_close(y[..., i*4:(i+1)*4],
                F.layer_norm(x[..., i*4:(i+1)*4], (4,), eps=1e-5))
        torch.testing.assert_close(y[..., -5:-1].mean((0, 1)), torch.zeros(4), atol=2e-6, rtol=0)
        torch.testing.assert_close(y[..., -5:-1].std((0, 1), unbiased=False), torch.ones(4))
        torch.testing.assert_close(y[..., -1], x[..., -1], rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)
        self.assertEqual(y.dtype, torch.float32)
        self.assertFalse(list(m.parameters()))

    def test_fit_train_only_once_and_no_inference_update(self):
        x = features()
        m = ObservationNormalizer(17)
        for split in ('validation', 'test'):
            with self.assertRaises(ValueError):
                m.fit(x, source_split=split)
        with self.assertRaises(ValueError):
            m(x)
        m.fit(x, source_split='train')
        state = copy.deepcopy(m.state_dict())
        m(features()*100)
        for key, value in state.items():
            torch.testing.assert_close(m.state_dict()[key], value, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            m.fit(x, source_split='train')

    def test_batch_and_checkpoint_independence(self):
        x = features()
        m = ObservationNormalizer(17).fit(x, source_split='train')
        y = m(x)
        torch.testing.assert_close(y[:1, :1], m(x[:1, :1]), rtol=0, atol=0)
        changed = x.clone()
        changed[1:] *= 100
        changed[0, 1:] *= 100
        torch.testing.assert_close(m(changed)[0, 0], y[0, 0], rtol=0, atol=0)

    def test_constant_and_zero_delta_are_finite(self):
        x = torch.ones(4, 7, 17)
        x[..., 8:12] = 0
        m = ObservationNormalizer(17).fit(x, source_split='train')
        y = m(x)
        self.assertTrue(torch.isfinite(y).all())
        torch.testing.assert_close(y[..., :12], torch.zeros_like(y[..., :12]))
        torch.testing.assert_close(m.phi_scale, torch.ones(4))

    def test_invalid_shape_nonfinite_or_corrupt_scale(self):
        with self.assertRaises(ValueError):
            ObservationNormalizer(16)
        x = features()
        m = ObservationNormalizer(17).fit(x, source_split='train')
        with self.assertRaises(ValueError):
            m(torch.full_like(x, float('nan')))
        with self.assertRaises(ValueError):
            m(torch.zeros(2, 7, 16))
        m.phi_scale[0] = 0
        with self.assertRaises(ValueError):
            m(x)


class NormalizedArchitectureTests(unittest.TestCase):
    def test_initial_weights_and_parameter_count_preserved(self):
        torch.manual_seed(42)
        raw = PPOController(17, 8)
        torch.manual_seed(42)
        new = NormalizedPPOController(17, 8)
        self.assertEqual(sum(p.numel() for p in raw.parameters()), sum(p.numel() for p in new.parameters()))
        for key, value in raw.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        self.assertIsInstance(new.encoder[1], nn.LayerNorm)
        self.assertFalse(new.encoder[1].elementwise_affine)
        self.assertEqual(new.encoder[0].in_features, 17)

    def test_negative_projection_is_centered_before_gelu(self):
        x = features()
        m = model_with_stats(x)
        with torch.no_grad():
            m.encoder[0].bias.fill_(-50)
            u = m.encoder[0](m.normalizer(x))
            centered = m.encoder[1](u)
        self.assertTrue((u < 0).all())
        torch.testing.assert_close(centered.mean(-1), torch.zeros_like(centered[..., 0]), atol=3e-5, rtol=0)
        self.assertTrue((centered > 0).any(-1).all())
        self.assertLess(encoder_diagnostics(m, x)['encoder_all_zero_fraction'], .01)

    def test_warmstart_transfer_has_same_stats_and_new_critic(self):
        x = features()
        warm = NormalizedWarmStartController(17, 8)
        warm.normalizer.fit(x, source_split='train')
        m = NormalizedPPOController(17, 8)
        m.load_warmstart(warm)
        actor, _ = warm(x)
        logits, value = m(x)
        torch.testing.assert_close(logits, actor, rtol=0, atol=0)
        torch.testing.assert_close(value, torch.full_like(value, .5))
        self.assertIsNot(m.normalizer.phi_mean, warm.normalizer.phi_mean)
        with self.assertRaises(ValueError):
            NormalizedPPOController(17, 8, projection_norm=False).load_warmstart(warm)

    def test_checkpoint_roundtrip_and_reject_legacy_or_wrong_architecture(self):
        x = features()
        m = model_with_stats(x)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pt'
            torch.save(dict(model=m.state_dict(), iteration=3), path)
            loaded = load_normalized_checkpoint(path)
            torch.testing.assert_close(loaded(x)[0], m(x)[0], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            m.load_state_dict(PPOController(17, 8).state_dict(), strict=False)
        with self.assertRaises(ValueError):
            NormalizedPPOController(17, 8, projection_norm=False).load_state_dict(m.state_dict())
        with self.assertRaises(ValueError):
            NormalizedPPOController(17, 8, eps=1e-4).load_state_dict(m.state_dict())

    def test_synthetic_warm_and_ppo_gradients_are_finite(self):
        x = features()
        warm = NormalizedWarmStartController(17, 8)
        warm.normalizer.fit(x, source_split='train')
        actor, q = warm(x)
        loss, _ = warmstart_loss(actor, q, torch.zeros(8, 7, dtype=torch.long), torch.ones_like(q))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in warm.parameters()))
        m = NormalizedPPOController(17, 8)
        m.load_warmstart(warm)
        ep = sample_episodes(m, x, torch.ones(8, 8))
        loss, _ = ppo_loss(m, x, ep, torch.zeros(8, 7, dtype=torch.bool),
            dict(clip=.2, value_weight=.5, entropy_weight=.01, tie_weight=.1))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_online_observations_are_raw_and_normalized_once(self):
        h = torch.randn(81, 4).bfloat16()
        x = torch.stack([raw_observation(h, k) for k in range(10, 80, 10)])[None]
        m = model_with_stats(x)
        expected = int(greedy_budgets(m, x)[0])
        stopper = OnlineStopper(m)
        stopper.begin(h[0])
        for k in range(1, 81):
            if stopper(step=k, hidden_last_position=h[k][None]):
                break
        self.assertEqual(k, expected)

    def test_projection_norm_can_be_disabled_for_ablation(self):
        x = features()
        m = model_with_stats(x, projection_norm=False)
        self.assertEqual(len(m.encoder), 2)
        torch.testing.assert_close(m(x)[0], m.actor(m.encoder(m.normalizer(x))))

    def test_batch_online_adapter_matches_offline_normalization(self):
        h = torch.randn(16, 81, 4).bfloat16()
        grid = torch.stack([torch.stack([raw_observation(v, k) for k in range(10, 80, 10)]) for v in h])
        m = model_with_stats(grid)
        expected = greedy_budgets(m, grid)
        online = CohortDecisions(m, 16, 'cpu', False)
        online.begin(h[:, 0])
        active = torch.ones(16, dtype=torch.bool)
        for k in range(1, 81):
            active &= ~online.step(k, h[:, k], active)
            if not active.any():
                break
        torch.testing.assert_close(online.budgets, expected, rtol=0, atol=0)
        torch.testing.assert_close(online.x[online.visited], grid[online.visited], rtol=0, atol=0)

    def test_logged_ppo_updates_keep_statistics_frozen(self):
        x = features(32)
        m = model_with_stats(x)
        before = copy.deepcopy(m.normalizer.state_dict())
        episode = sample_episodes(m, x, torch.ones(32, 8))
        cfg = dict(optimization_epochs=4, minibatch_questions=8, clip=.2,
                   value_weight=.5, entropy_weight=.01, tie_weight=.1, grad_clip=1.)
        optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)
        logs = []
        loss, updates = logged_ppo_update(m, optimizer, x, episode, x,
            torch.ones(32, 7, dtype=torch.bool), cfg, lambda step, row: logs.append(row))
        self.assertEqual(updates, 16)
        self.assertEqual(len(logs), 16)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        for key, value in before.items():
            torch.testing.assert_close(m.normalizer.state_dict()[key], value, rtol=0, atol=0)

    def test_new_config_is_separate_and_default_is_not_training(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root/'configs/controller_ppo_normalized.yaml')
        self.assertTrue(cfg['projection_norm'])
        self.assertIn('normalized', cfg['out'])
        self.assertEqual(cfg['iterations'], 80)


if __name__ == '__main__':
    unittest.main()
