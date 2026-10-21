import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from controller.matched import train_items,test_items,reference_prompt_audit,online_rollout


ROOT=Path(__file__).resolve().parents[1]


class ForcedDecisions:
    def __init__(self,budgets):
        self.budgets=torch.tensor(budgets)
        self.histories=[[] for _ in budgets]
    def begin(self,h):
        for i in range(len(h)):self.histories[i].append(h[i].clone())
    def step(self,k,h,active):
        for i in torch.where(active)[0].tolist():self.histories[i].append(h[i].clone())
        return active&(self.budgets.to(h.device)==k)


class MatchedTests(unittest.TestCase):
    def test_training_entry_rejects_test_even_if_caller_bypasses_loader(self):
        from scripts.run_matched_controller_ppo import require_train
        for split in ('test','validation'):
            bad=[dict(source_split=split,source_id=f'medqa:{split}:{i}') for i in range(32)]
            with self.assertRaises(ValueError):require_train(bad)
    def test_train_firewall_real_exports(self):
        items,manifest=train_items(ROOT/'data/medqa_usmle_4options')
        self.assertEqual(len(items),32)
        self.assertTrue(all(i['source_split']=='train' and ':train:' in i['source_id'] for i in items))
        self.assertFalse(manifest['heldout_overlap'])
        self.assertTrue(all(i['question'].endswith('Choose the correct option.') for i in items))
        with self.assertRaises(ValueError):train_items(ROOT/'data/medqa_usmle_4options',16)

    def test_test_cohort_identity_exact_question(self):
        ref=ROOT/'output/latent_mas_Qwen_Qwen3-4B_medqa_20260930_035619_469750.json'
        items=test_items(ROOT/'data/medqa_usmle_4options',ref)
        self.assertEqual([i['historical_index'] for i in items],list(range(16)))
        self.assertTrue(all(i['source_split']=='test' for i in items))
        train,_=train_items(ROOT/'data/medqa_usmle_4options')
        self.assertFalse({i['source_id'] for i in train}&{i['source_id'] for i in items})

    def test_cached_tokenizer_historical_prompt_exact(self):
        from transformers import AutoTokenizer
        from models import ModelWrapper
        tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B',revision='1cfa9a7208912126459214e8b04321603b3df60c',local_files_only=True)
        tokenizer.padding_side='left'
        wrapper=ModelWrapper.__new__(ModelWrapper);wrapper.tokenizer=tokenizer
        args=SimpleNamespace(task='medqa',think=False,model_name='Qwen/Qwen3-4B')
        report=reference_prompt_audit(wrapper,args,ROOT/'data/medqa_usmle_4options',ROOT/'output/latent_mas_Qwen_Qwen3-4B_medqa_20260930_035619_469750.json')
        self.assertTrue(report['passed'])

    def test_batch_padding_positions_and_prefixes(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        from latent_batch import rollout
        torch.manual_seed(42)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=40,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)).eval()
        wrapper=SimpleNamespace(HF_model=model,_apply_latent_realignment=lambda h,_:h)
        ids=torch.tensor([[0,0,3,4],[0,5,6,7],[8,9,10,11]])
        _,fixed=rollout(wrapper,ids,ids.ne(0),latent_steps=30)
        for preserve in (False,True):
            decisions=ForcedDecisions([10,20,30])
            past,prefixes,physical=online_rollout(wrapper,ids,ids.ne(0),decisions,preserve_arithmetic=preserve)
            self.assertEqual(physical,30)
            self.assertEqual([len(p) for p in prefixes],[12,23,34])
            self.assertEqual([len(h) for h in decisions.histories],[11,21,31])
            for i,k in enumerate([10,20,30]):torch.testing.assert_close(prefixes[i],fixed[i][:int(ids[i].ne(0).sum())+k],rtol=1e-5,atol=1e-6)
            self.assertEqual(past._latent_next_position.tolist(),[32,33,34] if preserve else [12,23,34])
            self.assertEqual(past._latent_attention_mask.sum(-1).tolist(),[32,33,34] if preserve else [12,23,34])


if __name__=='__main__':unittest.main()
