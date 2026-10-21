from collections import Counter
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from controller.matched import train_items
from controller.sampler import train_pool, ShuffledQuestionSampler
from controller.profile import normalized_question, load_official_medqa, write_json, write_jsonl
from controller.ppo import PPOController, sample_episodes
from scripts import run_pooled_controller_ppo as pooled
from scripts.run_ppo_with_wandb import PPOLoggingSession

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/medqa_usmle_4options'
CONFIG = dict(optimization_epochs=4, minibatch_questions=8, clip=.2,
              value_weight=.5, entropy_weight=.01, tie_weight=.1, grad_clip=1)


class SamplerTests(unittest.TestCase):
    def test_real_pool256_contains_rehearsal_and_excludes_heldout(self):
        old, _ = train_items(DATA)
        old_ids = [i['source_id'] for i in old]
        items, manifest = train_pool(DATA, required_ids=old_ids)
        repeated, repeat_manifest = train_pool(DATA, required_ids=old_ids)
        self.assertEqual(manifest, repeat_manifest)
        self.assertEqual(items, repeated)
        self.assertEqual(len(items), 256)
        self.assertEqual(len(set(manifest['source_ids'])), 256)
        self.assertEqual(manifest['source_ids'][:32], old_ids)
        stems = {normalized_question(i['question_stem']) for i in items}
        self.assertEqual(len(stems), 256)
        for split in ('validation', 'test'):
            heldout, _ = load_official_medqa(DATA, split)
            self.assertFalse(stems & {normalized_question(i['question_stem']) for i in heldout})
        self.assertTrue(all(i['source_split'] == 'train' and ':train:' in i['source_id'] for i in items))
        self.assertTrue(all(i['question'].endswith('Choose the correct option.') for i in items))

    def test_eight_iterations_cover256_once_and_next_epoch_reshuffles(self):
        ids = [f'train:{i}' for i in range(256)]
        sampler = ShuffledQuestionSampler(ids)
        first = [sampler.plan(i) for i in range(1, 9)]
        second = [sampler.plan(i) for i in range(9, 17)]
        for plans in (first, second):
            flat = [s for p in plans for s in p['source_ids']]
            self.assertEqual(len(flat), 256)
            self.assertEqual(set(flat), set(ids))
            self.assertTrue(all(len(set(p['indices'])) == 32 for p in plans))
            self.assertTrue(all(p['cohort_sizes'] == [16, 16] for p in plans))
        self.assertNotEqual([p['indices'] for p in first], [p['indices'] for p in second])
        self.assertEqual(second[0]['epoch'], 2)
        self.assertEqual(second[0]['iteration_in_epoch'], 1)

    def test_resume_plan_is_pure_and_does_not_consume_policy_rng(self):
        ids = list(range(256))
        sampler = ShuffledQuestionSampler(ids)
        expected = sampler.plan(6)
        python_rng = random.getstate()
        torch_rng = torch.get_rng_state().clone()
        for i in (9, 1, 80, 3):
            sampler.plan(i)
        self.assertEqual(ShuffledQuestionSampler(ids).plan(6), expected)
        self.assertEqual(random.getstate(), python_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        self.assertNotEqual(ShuffledQuestionSampler(ids, 43).plan(6), expected)

    def test_invalid_pool_and_iteration_rejected(self):
        for count in (0, 16, 33, -32):
            with self.assertRaises(ValueError):
                train_pool(DATA, count)
        with self.assertRaises(ValueError):
            train_pool(DATA, 32, required_ids=['medqa:test:0'])
        with self.assertRaises(ValueError):
            ShuffledQuestionSampler(['duplicate'] * 32)
        for iteration in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                ShuffledQuestionSampler(list(range(32))).plan(iteration)

    def test_committed_progress_ignores_unfinished_and_rejects_changed_plan(self):
        sampler = ShuffledQuestionSampler(list(range(64)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = pooled.new_attempt(root / 'train/iteration001')
            write_json(path / 'sampler_plan.json', sampler.plan(1))
            write_json(path / 'result.json', dict(iteration=1, total_controller_iterations=8,
                train_only=True, episodes=32))
            (path / 'checkpoint.pt').write_bytes(b'fake checkpoint')
            pooled.commit(root / 'train/iteration001', path)
            pooled.new_attempt(root / 'train/iteration002')
            last, checkpoint = pooled.committed_progress(root, sampler, 4, 7)
            self.assertEqual(last, 1)
            self.assertEqual(checkpoint, path / 'checkpoint.pt')
            self.assertEqual(sampler.plan(last + 1)['iteration_in_epoch'], 2)
            with self.assertRaises(ValueError):
                pooled.committed_progress(root, ShuffledQuestionSampler(list(range(64)), 43), 4, 7)

    def test_mock_live_iteration_uses_selected_questions_and_updates_only_after_both_cohorts(self):
        items, split = train_pool(DATA, count=64)
        sampler = ShuffledQuestionSampler(split['source_ids'])
        plan = sampler.plan(2)
        selected = [items[i] for i in plan['indices']]
        torch.manual_seed(42)
        model = PPOController(9, 16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=0)
        x = torch.randn(32, 7, 9)
        e = sample_episodes(model, x, torch.randint(0, 2, (32, 8)))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        next_group, observed_ids = [], []

        def decisions(*args):
            group = len(next_group)
            next_group.append(group)
            sl = slice(group * 16, (group + 1) * 16)
            return SimpleNamespace(x=x[sl], actions=e['actions'][sl],
                logprobs=e['old_logprobs'][sl], values=e['old_values'][sl],
                visited=e['visited'][sl], budgets=torch.full((16,), 10),
                histories=[[torch.zeros(9)] for _ in range(16)])

        def render(wrapper, args, cohort):
            encoded = {'A': dict(input_ids=torch.ones(16, 1), attention_mask=torch.ones(16, 1))}
            return {}, encoded

        def decode(wrapper, args, cohort, payloads, budgets, root, path, purpose):
            # Collection must not train between its two ordered cohorts.
            self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
            self.assertEqual(purpose, 'live_train_reward')
            observed_ids.extend(i['source_id'] for i in cohort)
            rows = [dict(source_id=i['source_id'], strict_correct=True) for i in cohort]
            write_jsonl(path / 'outcomes.jsonl', rows)
            return rows, .1

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(pooled.torch.cuda, 'synchronize'), \
                patch.object(pooled.torch.cuda, 'get_rng_state_all', return_value=[]), \
                patch.object(pooled, 'reset_cache') as reset, \
                patch.object(pooled, 'render_batch', side_effect=render), \
                patch.object(pooled, 'CohortDecisions', side_effect=decisions), \
                patch.object(pooled, 'online_rollout', return_value=(None, [], 10)) as rollout, \
                patch.object(pooled, 'prefixes_for_b', return_value=[]), \
                patch.object(pooled, 'decode_cohort', side_effect=decode):
            root = Path(directory)
            path = pooled.run_iteration(None, SimpleNamespace(device='cpu', device2='cpu'), root,
                selected, model, optimizer, x.clone(), torch.zeros(32, 7, dtype=torch.bool), CONFIG, plan, 7)
            result = json.loads((path / 'result.json').read_text())
            saved = torch.load(path / 'checkpoint.pt', weights_only=True)
            self.assertEqual(saved['iteration'], 9)
            self.assertEqual(result['optimizer_updates'], 16)
            self.assertEqual(observed_ids, plan['source_ids'])
            self.assertTrue(all(float(s['step']) == 16 for s in optimizer.state.values()))
            self.assertTrue(any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
            self.assertEqual(reset.call_count, 1)
            self.assertEqual(rollout.call_count, 2)
            self.assertTrue(all(c.kwargs['preserve_arithmetic'] for c in rollout.call_args_list))

    def test_pooled_logging_resume_counts_local_and_parent_coverage(self):
        torch.manual_seed(42)
        model = PPOController(9, 16)
        x = torch.randn(32, 7, 9)
        e = sample_episodes(model, x, torch.randint(0, 2, (32, 8)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'manifest.json', dict(training='/parent', parent_config={}, parent_iteration=7,
                iterations=8, train_manifest={'count': 256}, parent_train_ids=list(range(32)),
                sampler=dict(seed=42, iterations_per_epoch=8)))
            marker = root / 'train/iteration001/complete.json'
            marker.parent.mkdir(parents=True)
            marker.write_text('{}')
            opt = SimpleNamespace(wandb_mode='disabled', wandb_project='test', wandb_entity=None, wandb_name=None)
            session = PPOLoggingSession(root, opt, 'pooled')
            def coverage(path, through):
                if str(path) == '/parent':
                    return set(range(32)), 224
                self.assertEqual(through, 1)
                return set(range(16, 48)), 32
            with patch('scripts.run_ppo_with_wandb.prior_coverage', side_effect=coverage):
                session.update(model, optimizer, x, e, x, torch.zeros(32, 7, dtype=torch.bool), CONFIG)
            session.event(lambda *a, **k: None, root, stage='PPO_iteration_complete', iteration=9,
                correct=20, seconds=1., mean_K=10., sampler_epoch=1, iteration_in_epoch=2, pool_size=256)
            session.close(0)
            rows = list(map(json.loads, (root / 'training_metrics.jsonl').read_text().splitlines()))
            self.assertEqual(rows[0]['optimizer_step'], 129)
            self.assertEqual(rows[-1]['data/train_pool_size'], 256)
            self.assertEqual(rows[-1]['data/iteration_in_epoch'], 2)
            self.assertEqual(session.episodes_seen, 256)
            self.assertEqual(len(session.seen_ids), 48)


class SamplerEpochCoverageTests(unittest.TestCase):
    def test_two_epochs_revisit_each_question_once_per_epoch(self):
        ids=['m:train:'+str(i) for i in range(1024)]
        sampler=ShuffledQuestionSampler(ids,42)
        first=[sid for n in range(1,33) for sid in sampler.plan(n)['source_ids']]
        second=[sid for n in range(33,65) for sid in sampler.plan(n)['source_ids']]
        self.assertEqual(Counter(first),Counter(ids))
        self.assertEqual(Counter(second),Counter(ids))
        self.assertNotEqual(first,second)
        self.assertEqual(sampler.plan(64)['epoch'],2)


if __name__ == '__main__':
    unittest.main()
