import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from controller.profile import write_json, write_jsonl, write_tensors
from controller.matched import train_items
from scripts import profile_controller_pool as runner

ROOT = Path(__file__).resolve().parents[1]


class PoolProfileTests(unittest.TestCase):
    def test_real_selection_matches_approved_sampler_and_separates_validation(self):
        old, _ = train_items(ROOT / 'data/medqa_usmle_4options')
        items, selections = runner.select_splits(ROOT / 'data/medqa_usmle_4options', [i['source_id'] for i in old])
        self.assertEqual(len(items['train']), 256)
        self.assertEqual(len(items['validation']), 128)
        self.assertEqual(selections['train']['source_ids'][:32], [i['source_id'] for i in old])
        self.assertFalse(set(selections['train']['source_ids']) & set(selections['validation']['source_ids']))
        for split in items:
            runner.checked_cohort(items[split][:16], split)
        with self.assertRaises(ValueError):
            runner.checked_cohort(items['validation'][:16], 'train')
        with self.assertRaises(ValueError):
            runner.checked_cohort([dict(source_id='x:test:0', source_split='test')] * 16, 'test')

    def test_resume_cache_replays_all_committed_cohorts_and_no_cross_split_labels(self):
        # Three cohorts: resume must replay TWO prior cohorts, not only the first.
        items = [dict(source_id=f'x:validation:{i}', source_split='validation') for i in range(48)]
        teacher = dict(A_mask=torch.ones(16, 2), **{f'prefix_{i}': torch.ones(82, 2) for i in range(16)})
        encoded = {'A': dict(attention_mask=torch.ones(16, 2))}
        purposes = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for g in range(3):
                base = root / 'validation/teachers' / f'cohort{g:03d}'
                path = runner.new_attempt(base)
                write_json(path / 'cohort.json', dict(source_ids=[i['source_id'] for i in items[g * 16:(g + 1) * 16]]))
                runner.commit(base, path)
            write_json(root / 'validation/prefix_parity.json', {'passed': True})
            for g in range(2):
                base = root / 'validation/profile/K10' / f'cohort{g:03d}'
                path = runner.new_attempt(base)
                write_json(path / 'result.json', {'complete': True})
                runner.commit(base, path)
            def decode(wrapper, args, cohort, payloads, budgets, root, path, purpose):
                purposes.append(purpose)
                rows = [dict(source_id=i['source_id'], source_split='validation', strict_correct=True)
                        for i in cohort]
                write_jsonl(path / 'outcomes.jsonl', rows)
                return rows, .1
            with patch.object(runner, 'read_teacher', return_value=(teacher, {}, encoded)), \
                    patch.object(runner, 'prefixes_for_b', return_value=[]), \
                    patch.object(runner, 'reset_cache') as resets, \
                    patch.object(runner, 'decode_cohort', side_effect=decode):
                runner.profile_split(None, SimpleNamespace(device2='cpu'), items, root, 'validation')
            self.assertEqual(purposes[:3], ['validation_profile_cache_replay'] * 2 + ['validation_full_grid'])
            self.assertEqual(purposes.count('validation_full_grid'), 22)
            self.assertEqual(resets.call_count, 8)
            self.assertEqual(len(list(root.glob('validation/cache_replay/K10/cohort*/attempt_*/replay_complete.json'))), 2)

    def test_export_train_and_validation_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'validation'):
                items = [dict(source_id=f'x:{split}:{i}', source_split=split) for i in range(16)]
                base = root / split / 'teachers/cohort000'
                path = runner.new_attempt(base)
                write_tensors(path / 'feature_grid.safetensors', {'features': torch.zeros(16, 8, 9)})
                runner.commit(base, path)
                for k in range(10, 81, 10):
                    base = root / split / 'profile' / f'K{k}/cohort000'
                    path = runner.new_attempt(base)
                    rows = [dict(source_id=i['source_id'], source_split=split, k=k, strict_correct=True) for i in items]
                    write_jsonl(path / 'outcomes.jsonl', rows)
                    runner.commit(base, path)
                dataset = runner.export_dataset(root, items, split)
                tensors = load_file(str(dataset / 'dataset.safetensors'))
                self.assertEqual(tensors['features'].shape, (16, 7, 9))
                if split == 'train':
                    self.assertTrue(tensors['actions'].eq(0).all())
                    self.assertEqual(tensors['q_targets'].shape, (16, 7, 2))
                else:
                    self.assertEqual(set(tensors), {'features', 'correctness'})
                self.assertEqual(json.loads((dataset / 'manifest.json').read_text())['allowed_for_gradient_updates'], split == 'train')
                self.assertEqual(runner.export_dataset(root, items, split), dataset)


if __name__ == '__main__':
    unittest.main()
