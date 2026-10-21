import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from controller.profile import write_json, atomic_exclusive_bytes
from scripts.run_matched_controller_ppo import commit
from scripts.evaluate_expanded_controller_test300 import (
    cohort_items, match_items, selected_checkpoint, decode_test_cohort)


class ExpandedTest300Tests(unittest.TestCase):
    def test_matches_embedded_newline_options_and_rejects_duplicate_or_wrong_gold(self):
        options = ['A. one', 'B. two', 'C. three', 'D. four\n"']
        historical = dict(question='Question\n'+'\n'.join(options), options=options, answer='one')
        official = dict(source_id='m:test:1', source_split='test', question_stem='Question',
            question='Question\n\n'+'\n'.join(options), gold='a')
        reference = dict(question='Question\n'+'\n'.join(options)+'\nChoose the correct option.', gold='a')
        items = match_items([historical], [official], [reference])
        self.assertEqual(items[0]['source_id'], 'm:test:1')
        self.assertEqual(items[0]['question'], reference['question'])
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            match_items([historical]*2, [official], [reference]*2)
        bad = copy.deepcopy(historical); bad['answer'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'gold mismatch'):
            match_items([bad], [official], [reference])
        bad_reference = dict(reference, question='different prompt')
        with self.assertRaisesRegex(ValueError, 'reference mismatch'):
            match_items([historical], [official], [bad_reference])

    def test_all300_ordered_once_with_unpadded_last12(self):
        items = list(range(300))
        groups = cohort_items(items)
        self.assertEqual([len(g) for g in groups], [16]*18+[12])
        self.assertEqual([i for group in groups for i in group], items)

    def test_requested_validation_checkpoint_is_verified_and_test_trained_run_rejected(self):
        from controller.profile import file_hash
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_json(root/'training_finished.json', dict(status='complete', test300_used=False,
                best_validation=dict(iteration=16)))
            base = root/'train/iteration016'; attempt = base/'attempt_000'; attempt.mkdir(parents=True)
            write_json(attempt/'result.json', dict(iteration=16, train_only=True, episodes=32))
            atomic_exclusive_bytes(attempt/'checkpoint.pt', b'frozen checkpoint')
            commit(base, attempt)
            digest = file_hash(attempt/'checkpoint.pt')
            self.assertEqual(selected_checkpoint(root, 16, digest), attempt/'checkpoint.pt')
            with self.assertRaises(ValueError): selected_checkpoint(root, 32, digest)
            with self.assertRaises(ValueError): selected_checkpoint(root, 16, 'wrong')
            (attempt/'checkpoint.pt').write_bytes(b'corrupted')
            with self.assertRaises(ValueError): selected_checkpoint(root, 16, digest)

    def test_real_partial_decoder_counts12_and_rejects_short_engine_output(self):
        fake_vllm = SimpleNamespace(SamplingParams=lambda **kw: kw)
        items = [dict(source_id=f'm:test:{i}', source_split='test', historical_index=i, gold='a') for i in range(12)]
        payloads = [dict(prompt_embeds=torch.zeros(3,4)) for _ in items]
        outputs = [SimpleNamespace(outputs=[SimpleNamespace(text='Answer: \\boxed{A}',
            token_ids=[1,2,3], finish_reason='stop', stop_reason=None)], num_cached_tokens=0) for _ in items]
        class Engine:
            def generate(self, payloads, params, use_tqdm=False):
                self.params = params
                return outputs
        engine = Engine(); wrapper = SimpleNamespace(vllm_engine=engine)
        with tempfile.TemporaryDirectory() as td, patch.dict(sys.modules, {'vllm':fake_vllm}):
            root = Path(td)
            rows, _ = decode_test_cohort(wrapper, None, items, payloads, [10]*12, root, root)
            self.assertEqual(len(rows),12)
            self.assertEqual(sum(r['strict_correct'] for r in rows),12)
            self.assertTrue(all(r['batch']==12 for r in rows))
            self.assertEqual(engine.params['seed'],42)
            self.assertEqual(engine.params['temperature'],.6)
            self.assertEqual(len((root/'outcomes.jsonl').read_text().splitlines()),12)
            outputs.pop()
            with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                decode_test_cohort(wrapper, None, items, payloads, [10]*12, root, root)


if __name__ == '__main__':
    unittest.main()
