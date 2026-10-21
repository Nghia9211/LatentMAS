from pathlib import Path
import unittest
import torch
from controller.sampler import fresh_train_items
from controller.matched import train_items
from controller.profile import normalized_question
from controller.ppo import PPOController, sample_episodes
from scripts.run_matched_controller_ppo import ppo_update

ROOT = Path(__file__).resolve().parents[1]


class FreshIterationTests(unittest.TestCase):
    def test_real_fresh_train_disjoint_and_reproducible(self):
        old, _ = train_items(ROOT / 'data/medqa_usmle_4options')
        old_ids = [i['source_id'] for i in old]
        items, manifest = fresh_train_items(ROOT / 'data/medqa_usmle_4options', old_ids)
        repeated, _ = fresh_train_items(ROOT / 'data/medqa_usmle_4options', old_ids)
        self.assertEqual(len(items), 32)
        self.assertEqual([i['source_id'] for i in repeated], manifest['source_ids'])
        self.assertFalse(set(old_ids) & set(manifest['source_ids']))
        self.assertFalse({normalized_question(i['question_stem']) for i in old} &
                         {normalized_question(i['question_stem']) for i in items})
        self.assertTrue(all(i['source_split'] == 'train' and ':train:' in i['source_id'] for i in items))
        self.assertTrue(all(i['question'].endswith('Choose the correct option.') for i in items))

    def test_parent_ids_must_be_official_train(self):
        with self.assertRaises(ValueError):
            fresh_train_items(ROOT / 'data/medqa_usmle_4options', ['medqa:test:0'] * 32)

    def test_four_epochs_sixteen_updates_and_frozen_rollout(self):
        torch.manual_seed(42)
        model = PPOController(9, 16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=0)
        x = torch.randn(32, 7, 9)
        episode = sample_episodes(model, x, torch.randint(0, 2, (32, 8)))
        old_logprobs = episode['old_logprobs'].clone()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        teacher_x = torch.randn_like(x)  # Separate fixed rehearsal, not fresh labels.
        config = dict(optimization_epochs=4, minibatch_questions=8, clip=.2,
                      value_weight=.5, entropy_weight=.01, tie_weight=.1, grad_clip=1)
        loss, updates = ppo_update(model, optimizer, x, episode, teacher_x,
                                   torch.ones(32, 7, dtype=torch.bool), config)
        self.assertEqual(updates, 16)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(torch.equal(old_logprobs, episode['old_logprobs']))
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        self.assertTrue(all(float(s['step']) == 16 for s in optimizer.state.values()))


if __name__ == '__main__':
    unittest.main()
