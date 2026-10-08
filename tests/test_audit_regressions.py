"""CPU regressions for the upstream audit; no pretrained weights needed."""
import argparse
import contextlib
import io
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from models import ModelWrapper
from methods.latent_mas import LatentMASMethod
from methods.baseline import BaselineMethod
from methods.text_mas import TextMASMethod
from prompts import build_agent_messages_sequential_text_mas
from data import load_medqa
import run
from code_execution import run_with_timeout


class Tokenizer:
    pad_token_id = 0
    def decode(self, ids, **kwargs):
        return ','.join(str(i) for i in ids.tolist())


class HFAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.model = Qwen3ForCausalLM(Qwen3Config(vocab_size=40, hidden_size=32,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, eos_token_id=None, pad_token_id=0)).eval()

    def wrapper(self):
        w = ModelWrapper.__new__(ModelWrapper)
        w.model = self.model
        w.device = torch.device('cpu')
        w.tokenizer = Tokenizer()
        w._apply_latent_realignment = lambda h, _: h
        return w

    def test_hf_greedy_and_padding_match_single(self):
        w = self.wrapper()
        for short in ([4, 5, 0], [0, 4, 5]):
            with self.subTest(short=short):
                ids = torch.tensor([short, [6, 7, 8]])
                texts, _ = w.generate_text_batch(ids, ids.ne(0), temperature=0, max_new_tokens=3)
                for i in range(2):
                    single = ids[i, ids[i].ne(0)][None]
                    expected, _ = w.generate_text_batch(single, temperature=0, max_new_tokens=3)
                    self.assertEqual(texts[i], expected[0])
                    self.assertEqual(len(texts[i].split(',')), 3)

    def test_hf_latent_padding_and_metadata(self):
        w = self.wrapper()
        for short in ([4, 5, 0], [0, 4, 5]):
            ids = torch.tensor([short, [6, 7, 8]])
            cache = w.generate_latent_batch(ids, ids.ne(0), latent_steps=2)
            self.assertEqual(cache._latent_attention_mask.sum(-1).tolist(), [4, 5])
            other = torch.tensor([[0, 9, 10], [11, 12, 13]])
            batch, _ = w.generate_text_batch(other, other.ne(0), past_key_values=cache,
                                              temperature=0, max_new_tokens=3)
            for i in range(2):
                single = ids[i, ids[i].ne(0)][None]
                c = w.generate_latent_batch(single, latent_steps=2)
                text, _ = w.generate_text_batch(other[i, other[i].ne(0)][None],
                    past_key_values=c, temperature=0, max_new_tokens=3)
                self.assertEqual(batch[i], text[0])

    def test_hf_rejects_negative_steps(self):
        with self.assertRaises(ValueError):
            self.wrapper().generate_latent_batch(torch.tensor([[4, 5]]), latent_steps=-1)

    def test_empty_cache_greedy(self):
        from transformers.cache_utils import DynamicCache
        w = self.wrapper()
        ids = torch.tensor([[4, 5]])
        first, _ = w.generate_text_batch(ids, past_key_values=DynamicCache(), temperature=0, max_new_tokens=3)
        second, _ = w.generate_text_batch(ids, temperature=0, max_new_tokens=3)
        self.assertEqual(first, second)

    def test_trimmed_cache_keeps_positions(self):
        w = self.wrapper()
        ids = torch.tensor([[4, 5, 0], [6, 7, 8]])
        method = LatentMASMethod.__new__(LatentMASMethod)
        cache = w.generate_latent_batch(ids, ids.ne(0), latent_steps=2)
        cache = method._truncate_past(cache, 2)
        other = torch.tensor([[9], [10]])
        if not hasattr(w.model, '_prepare_position_ids_for_generation'):
            with self.assertRaisesRegex(ValueError, 'Transformers 5.x'):
                w.generate_text_batch(other, past_key_values=cache, temperature=0, max_new_tokens=3)
            return
        batch, _ = w.generate_text_batch(other, past_key_values=cache, temperature=0, max_new_tokens=3)
        for i in range(2):
            single = ids[i, ids[i].ne(0)][None]
            c = method._truncate_past(w.generate_latent_batch(single, latent_steps=2), 2)
            text, _ = w.generate_text_batch(other[i:i+1], past_key_values=c, temperature=0, max_new_tokens=3)
            self.assertEqual(batch[i], text[0])


class ScoringAndDataTests(unittest.TestCase):
    def test_dual_medqa_scores_for_text_and_baseline(self):
        ids = torch.tensor([[1, 2]])
        model = SimpleNamespace(use_vllm=True,
            prepare_chat_batch=lambda *a, **kw: (['prompt'], ids, ids.ne(0), [['a','b']]),
            vllm_generate_text_batch=lambda *a, **kw: [r'The answer is \boxed{B}'])
        args = SimpleNamespace(task='medqa', num_agents=2, prompt='sequential')
        item = dict(question='q', gold='b', solution='b')
        for cls, builder in [(BaselineMethod, 'methods.baseline.build_agent_messages_single_agent'),
                             (TextMASMethod, 'methods.text_mas.build_agent_messages_sequential_text_mas')]:
            method = cls(model, args=args, **({'use_vllm': True} if cls is BaselineMethod else {}))
            with patch(builder, return_value=[]):
                result = method.run_batch([item])[0]
            self.assertTrue(result['correct'])
            self.assertTrue(result['medqa_boxed_v1']['correct'])

    def test_textmas_unlimited_context_keeps_last_character(self):
        args = SimpleNamespace(model_name='Qwen/tiny', task='medqa', text_mas_context_length=-1)
        messages = build_agent_messages_sequential_text_mas(role='judger', question='q',
            context='UNIQUE_CONTEXT_END#', method='text_mas', args=args)
        self.assertIn('UNIQUE_CONTEXT_END#', str(messages))

    def test_missing_aime_answer_is_incorrect_not_exception(self):
        ids = torch.tensor([[1, 2]])
        model = SimpleNamespace(use_vllm=True,
            prepare_chat_batch=lambda *a, **kw: (['prompt'], ids, ids.ne(0), [['a','b']]),
            vllm_generate_text_batch=lambda *a, **kw: ['I cannot answer.'])
        args = SimpleNamespace(task='aime2024', num_agents=2, prompt='sequential')
        item = dict(question='q', gold='12', solution='12')
        for cls, builder in [(BaselineMethod, 'methods.baseline.build_agent_messages_single_agent'),
                             (TextMASMethod, 'methods.text_mas.build_agent_messages_sequential_text_mas')]:
            method = cls(model, args=args, **({'use_vllm': True} if cls is BaselineMethod else {}))
            with patch(builder, return_value=[]):
                self.assertFalse(method.run_batch([item])[0]['correct'])

    def test_medqa_missing_answer_does_not_reuse_previous_gold(self):
        rows = [dict(query='q1', answer='one', options=['A. one','B. two','C. three','D. four']),
                dict(query='q2', answer='missing', options=['A. one','B. two','C. three','D. four'])]
        with patch('data.load_dataset', return_value=rows):
            with self.assertRaises(ValueError):
                list(load_medqa())

    def test_medqa_answer_uses_exact_not_substring_match(self):
        rows = [dict(query='q', answer='one', options=['A. one plus','B. one','C. three','D. four'])]
        with patch('data.load_dataset', return_value=rows):
            self.assertEqual(list(load_medqa())[0]['gold'], 'b')

    def test_medqa_rejects_fake_training_split(self):
        with self.assertRaisesRegex(ValueError, 'separate loader'):
            list(load_medqa(split='train'))

    def test_packaged_medqa_all_labels_unchanged(self):
        import json
        rows = json.loads((Path(__file__).parents[1] / 'data/medqa.json').read_text())
        with patch('data.load_dataset', return_value=rows):
            actual = list(load_medqa())
        self.assertEqual(len(actual), len(rows))
        for source, target in zip(rows, actual):
            old_label = next('abcd'[i] for i, option in enumerate(source['options'])
                             if source['answer'] in option)
            self.assertEqual(target['gold'], old_label)


class RunnerTests(unittest.TestCase):
    cli = ['--method', 'latent_mas', '--model_name', 'Qwen/Qwen3-4B']

    def test_bad_arguments_fail_before_loading(self):
        for extra in (['--generate_bs', '0'], ['--max_samples', '0'],
                      ['--latent_steps', '-1'], ['--temperature', 'nan'], ['--top_p', '2']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run.parse_args(self.cli + extra)

    def test_yaml_validation(self):
        for cfg in ({'generate_bs': True}, {'num_agents': 3}, {'temperature': 'oops'},
                    {'use_vllm': 'false'}, {'latent_step': 20}, {'method': 'typo'}):
            with self.subTest(cfg=cfg), patch('run._load_yaml_config', return_value=cfg), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run.parse_args(self.cli + ['--config', 'mock.yaml'])

    def test_yaml_boolean_can_be_disabled(self):
        with patch('run._load_yaml_config', return_value={'use_vllm': True, 'enable_prefix_caching': True}):
            args = run.parse_args(self.cli + ['--config', 'mock.yaml', '--no-use_vllm', '--no-enable_prefix_caching'])
            self.assertFalse(args.use_vllm)
            self.assertFalse(args.enable_prefix_caching)
        args = run.parse_args(self.cli + ['--use_vllm', '--no-enable_prefix_caching'])
        self.assertTrue(args.use_second_HF_model)
        self.assertFalse(args.enable_prefix_caching)

    def test_actual_split_and_sample_limit(self):
        args = run.parse_args(self.cli + ['--task', 'aime2024', '--max_samples', '100'])
        self.assertEqual(args.split, 'train')
        args.split = 'validation'
        with patch('run.load_aime2024', return_value=iter([{'gold':'1'}])) as loader:
            self.assertEqual(len(run.load_items(args)), 1)
            loader.assert_called_once_with(split='validation')

    def test_empty_dataset_rejected(self):
        with patch('run.load_gsm8k', return_value=iter([])):
            with self.assertRaisesRegex(ValueError, 'empty'):
                run.load_items(run.parse_args(self.cli))

    def test_missing_batch_results_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Expected 1'):
            run.process_batch(SimpleNamespace(run_batch=lambda _: []), [{}], 0, [], None, 1,
                              argparse.Namespace(method='baseline', use_vllm=False))

    def test_hf_only_import_without_vllm(self):
        import subprocess
        import sys
        probe = """import sys
sys.modules['vllm'] = None
import run
import models
assert not models._HAS_VLLM
try:
    models.ModelWrapper('unused', 'cpu', use_vllm=True)
except RuntimeError:
    pass
else:
    raise AssertionError('must not silently fall back to HF')
"""
        result = subprocess.run([sys.executable, '-c', probe], cwd=Path(__file__).parents[1],
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_latent_run_item_dispatch(self):
        method = LatentMASMethod.__new__(LatentMASMethod)
        method.model = SimpleNamespace(use_vllm=True)
        method.run_batch_vllm = Mock(return_value=[{'correct': True}])
        method.run_batch = Mock(side_effect=AssertionError('wrong backend'))
        self.assertTrue(method.run_item({'question': 'q'})['correct'])

    def test_summary_counts_real_samples(self):
        args = run.parse_args(['--method','baseline','--model_name','Qwen/Qwen3-4B','--max_samples','99'])
        method = SimpleNamespace(run_batch=lambda items: [{'correct':True, 'question':'q'} for _ in items])
        with patch('run.parse_args', return_value=args), patch('run.set_seed'), \
             patch('run.load_items', return_value=[{'question':'q'}]), patch('run.ModelWrapper'), \
             patch('run.BaselineMethod', return_value=method), patch('run.save_results') as save, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run.main()
        summary = save.call_args.args[1]
        self.assertEqual(summary['max_samples'], 1)
        self.assertEqual(summary['requested_max_samples'], 99)
        self.assertEqual(summary['run_config']['max_samples'], 99)
        self.assertEqual(summary['time_per_sample_sec'], summary['total_time_sec'])


class ExecutionTests(unittest.TestCase):
    def test_spawn_worker_and_failure(self):
        self.assertEqual(run_with_timeout('assert 2 + 2 == 4', 20), (True, None))
        ok, error = run_with_timeout('assert False', 20)
        self.assertFalse(ok)
        self.assertIn('AssertionError', error)

    def test_timeout_reaps_child(self):
        before = {p.pid for p in multiprocessing.active_children()}
        ok, error = run_with_timeout('import time; time.sleep(10)', 0.1)
        self.assertFalse(ok)
        self.assertIn('TimeoutError', error)
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)


if __name__ == '__main__':
    unittest.main()
