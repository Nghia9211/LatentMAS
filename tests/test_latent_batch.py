import unittest
from types import SimpleNamespace
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from latent_batch import rollout, insert_prefix
from utils import medqa_scoring_metadata


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=40, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8)).eval()
        cls.wrapper = SimpleNamespace(HF_model=model,
                                      _apply_latent_realignment=lambda h, _: h)

    def test_padding_matches_individual_rollouts(self):
        for side in ("left", "right"):
            for steps in (0, 2):
                with self.subTest(side=side, steps=steps):
                    short = [0, 0, 4, 5] if side == "left" else [4, 5, 0, 0]
                    ids = torch.tensor([short, [6, 7, 8, 9]])
                    cache, emb = rollout(self.wrapper, ids, ids.ne(0).long(), latent_steps=steps)
                    self.assertEqual([len(e) for e in emb], [2 + steps, 4 + steps])
                    for i in range(2):
                        _, single = rollout(self.wrapper, ids[i, ids[i].ne(0)][None], latent_steps=steps)
                        torch.testing.assert_close(emb[i], single[0], atol=2e-6, rtol=2e-5)
                    self.assertEqual(cache._latent_attention_mask.sum(-1).tolist(), [2+steps, 4+steps])

    def test_three_latent_agents_match_individual(self):
        from methods.latent_mas import LatentMASMethod
        for trim in (None, "segment", "latent"):
            with self.subTest(trim=trim):
                batched = None
                singles = [None, None]
                for step in range(3):
                    ids = torch.tensor([[4+step, 5, 0], [6, 7+step, 8]])
                    old_len = batched.get_seq_length() if batched is not None else 0
                    batched, emb = rollout(self.wrapper, ids, ids.ne(0).long(), latent_steps=2, past_key_values=batched)
                    for i in range(2):
                        old_single = singles[i].get_seq_length() if singles[i] is not None else 0
                        singles[i], single = rollout(self.wrapper, ids[i, ids[i].ne(0)][None], latent_steps=2, past_key_values=singles[i])
                        torch.testing.assert_close(emb[i], single[0], atol=3e-6, rtol=3e-5)
                        if trim:
                            count = 2 if trim == "latent" else singles[i].get_seq_length()-old_single
                            singles[i] = LatentMASMethod._truncate_past(LatentMASMethod.__new__(LatentMASMethod), singles[i], count)
                    if trim:
                        count = 2 if trim == "latent" else batched.get_seq_length()-old_len
                        batched = LatentMASMethod._truncate_past(LatentMASMethod.__new__(LatentMASMethod), batched, count)

    def test_ragged_vllm_inputs(self):
        emb = torch.arange(24).reshape(2, 4, 3).float()
        mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
        prefixes = [torch.ones(1, 3), torch.ones(2, 3)]
        inputs = insert_prefix(emb, mask, prefixes, [1, 1])
        self.assertEqual([len(x['prompt_embeds']) for x in inputs], [3, 5])
        torch.testing.assert_close(inputs[0]['prompt_embeds'], torch.cat([emb[0,:1], prefixes[0], emb[0,1:2]]))

    def test_embedding_rollout_matches_ids_and_preserves_inputs(self):
        ids = torch.tensor([[0, 4, 5], [6, 7, 8]])
        _, from_ids = rollout(self.wrapper, ids, ids.ne(0), latent_steps=2)
        supplied = self.wrapper.HF_model.get_input_embeddings()(ids).detach()
        original = supplied.clone()
        _, from_embeds = rollout(self.wrapper, inputs_embeds=supplied,
                                attention_mask=ids.ne(0), latent_steps=2)
        torch.testing.assert_close(supplied, original)
        for a, b in zip(from_ids, from_embeds):
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
        with self.assertRaises(ValueError):
            rollout(self.wrapper, ids, inputs_embeds=supplied, latent_steps=1)

    def test_medqa_parser(self):
        for text, expected in [(r'Answer: \boxed{ C }', 'c'), ('Patient is 34', None),
                               ('I think B', None), (r'\boxed{B} then \boxed{wrong}', None)]:
            row = medqa_scoring_metadata(text, 'c', '34', False)
            self.assertEqual(row['medqa_boxed_v1']['prediction'], expected)
            self.assertEqual(row['legacy_prediction'], '34')

    def test_hybrid_method_two_and_four_agents(self):
        from methods.latent_mas import LatentMASMethod
        class Tokenizer:
            def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=False):
                if isinstance(texts, str):
                    return {'input_ids': [3 + ord(c) % 30 for c in texts]}
                ids = [[3 + ord(c) % 30 for c in text] for text in texts]
                width = max(map(len, ids))
                return {'input_ids': torch.tensor([r+[0]*(width-len(r)) for r in ids]),
                        'attention_mask': torch.tensor([[1]*len(r)+[0]*(width-len(r)) for r in ids])}
            def convert_ids_to_tokens(self, ids):
                return list(map(str, ids))
        class Engine:
            def generate(self, prompts, params):
                self.lengths = [len(p['prompt_embeds']) for p in prompts]
                return [SimpleNamespace(outputs=[SimpleNamespace(text=r'\boxed{C}',
                        token_ids=[1,2,3], finish_reason='stop', stop_reason=None)]) for p in prompts]
        from unittest.mock import patch
        tokenizer, engine = Tokenizer(), Engine()
        wrapper = SimpleNamespace(HF_model=self.wrapper.HF_model,
            _apply_latent_realignment=self.wrapper._apply_latent_realignment,
            embedding_layer=self.wrapper.HF_model.get_input_embeddings(),
            HF_device='cpu', tokenizer=tokenizer, vllm_engine=engine,
            prepare_chat_batch=lambda messages, **kw: (messages, None, None, None))
        wrapper.generate_latent_batch_hidden_state = lambda *a, **kw: rollout(wrapper, *a, **kw)
        for count in (2, 4):
            args = SimpleNamespace(num_agents=count, device='cpu', device2='cpu',
                max_new_tokens=8, task='medqa', prompt='sequential', think=False,
                model_name='Qwen/tiny')
            with patch('methods.latent_mas.SamplingParams', SimpleNamespace):
                method = LatentMASMethod(wrapper, latent_steps=2, generate_bs=2, args=args)
            items = [{'question': q, 'gold':'c', 'solution':'c'} for q in ('x', 'longer')]
            with patch('methods.latent_mas.build_agent_message_sequential_latent_mas',
                       side_effect=lambda **kw: '<|im_start|>user\n'+kw['question']):
                results = method.run_batch_vllm(items)
            self.assertEqual(len(results), 2)
            self.assertEqual(len(results[0]['agents']), count)
            self.assertEqual(engine.lengths[1] - engine.lengths[0], count * 5)
            self.assertTrue(all(r['medqa_boxed_v1']['correct'] for r in results))
            self.assertEqual(results[0]['generation']['finish_reason'], 'stop')
            self.assertEqual(results[0]['judger_mode'], 'text')
            self.assertNotIn('judger_latent_steps', results[0])
            for trace in results[0]['agents']:
                if trace['role'] == 'judger':
                    self.assertNotIn('latent_steps', trace)
                    self.assertEqual(trace['output'], r'\boxed{C}')
                else:
                    self.assertEqual(trace['latent_steps'], 2)


if __name__ == '__main__':
    unittest.main()
