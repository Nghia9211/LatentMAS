import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from controller.expanded_dataset import validate_expanded_tensors
from scripts import profile_controller_expansion as expansion
from scripts.run_matched_controller_ppo import new_attempt,commit
from controller.profile import write_json

ROOT=Path(__file__).resolve().parents[1]


def features(n,dim=17):
    x=torch.randn(n,7,dim)
    x[...,-1]=torch.arange(1,8)/8
    return x


def metadata(n,split):
    return dict(split=split,source_ids=[f'medqa:{split}:{i}' for i in range(n)],
        allowed_for_gradient_updates=split=='train',features_are_raw=True,seeds=[42,43,44])


class ExpansionTests(unittest.TestCase):
    def test_actual_selection_retains_all256_and_excludes_heldout_stems(self):
        p=ROOT/'profiles/medqa_train256_valid128_temp06_bs16_20261015_v1/manifest.json'
        if not p.exists(): self.skipTest('No archived manifest in fixture checkout')
        pm=json.loads(p.read_text());ids=pm['selections']['train']['source_ids']
        items,meta=expansion.selected_pool(ROOT/'data/medqa_usmle_4options',ids,1024)
        self.assertEqual(len(items),1024);self.assertEqual(meta['source_ids'][:256],ids)
        self.assertEqual(len(set(meta['source_ids'][256:])),768)
        self.assertFalse(set(meta['source_ids'][256:])&set(ids))
        self.assertFalse(meta['heldout_question_overlap'])
        self.assertTrue(all(x['source_split']=='train' for x in items))

    def test_invalid_expansion_or_partial_cohort_rejected(self):
        for n in (16,32,40):
            with self.assertRaises(ValueError):
                expansion.selected_pool('unused',[f'x:train:{i}' for i in range(32)],n)

    def test_fraction_targets_and_shorter_budget_tie(self):
        x=features(4);q=torch.zeros(4,8,3,dtype=torch.int8)
        q[:,0,0]=1;q[:,1:,0:2]=1
        data=expansion.tensors_from_samples(x,q,'train')
        torch.testing.assert_close(data['correctness'][:,0],torch.full((4,),1/3))
        self.assertEqual(data['actions'][0].tolist(),[1,0,0,0,0,0,0])
        m=metadata(4,'train')
        validate_expanded_tensors(data,m,'train',m['source_ids'])

    def test_variable_size_loader_and_validation_has_no_optimizer_targets(self):
        for n in (16,64,1024):
            x=features(n);q=torch.randint(0,2,(n,8,3),dtype=torch.int8)
            for split in ('train','validation'):
                data=expansion.tensors_from_samples(x,q,split)
                m=metadata(n,split)
                validate_expanded_tensors(data,m,split,m['source_ids'])
                self.assertEqual(len(data['features']),n)
                self.assertEqual('q_targets' in data,split=='train')
        with self.assertRaises(ValueError):
            expansion.tensors_from_samples(features(16),torch.ones(16,8,3),'test')

    def test_validation_cannot_smuggle_training_labels_or_shuffle_source_ids(self):
        m=metadata(16,'validation')
        data=expansion.tensors_from_samples(features(16),torch.zeros(16,8,3),'validation')
        bad=dict(data,actions=torch.zeros(16,7))
        with self.assertRaises(ValueError): validate_expanded_tensors(bad,m,'validation',m['source_ids'])
        with self.assertRaises(ValueError): validate_expanded_tensors(data,m,'validation',list(reversed(m['source_ids'])))
        bad={k:v.clone() for k,v in data.items()};bad['features'][0,0,-1]=0.
        with self.assertRaises(ValueError): validate_expanded_tensors(bad,m,'validation',m['source_ids'])

    def test_parent_aggregate_requires_complete_grid_and_distinct_rows(self):
        ids=['m:train:1'];rows=[dict(source_id=ids[0],k=k,seeds=[42,43,44],correctness=[1,0,1]) for k in range(10,81,10)]
        q=expansion.samples_from_rows(rows,ids)
        self.assertEqual(q.shape,(1,8,3))
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError): expansion.samples_from_rows(bad,ids)

    def test_resume_does_not_change_commits_and_rejects_holes(self):
        with tempfile.TemporaryDirectory() as d:
            bases=[Path(d)/f'cohort{i}' for i in range(3)]
            for b in bases[:2]:
                p=new_attempt(b);write_json(p/'result.json',dict(valid=True));commit(b,p)
            before=(bases[0]/'complete.json').read_bytes()
            self.assertEqual(expansion.progress(bases),2)
            self.assertEqual((bases[0]/'complete.json').read_bytes(),before)
            with self.assertRaises(ValueError): expansion.progress([bases[0],bases[2],bases[1]])

    def test_binary_samples_only_and_no_silent_missing_values(self):
        for q in (torch.full((16,8,3),.5),torch.full((16,8,3),-1),torch.zeros(16,7,3)):
            with self.assertRaises(ValueError): expansion.tensors_from_samples(features(16),q,'train')

    def test_last_logits_keeps_hidden_states_and_KV_identical(self):
        torch.manual_seed(42)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=40,hidden_size=32,
            intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=2,head_dim=8)).eval()
        ids=torch.tensor([[2,3,4,5],[6,7,8,9]])
        with torch.no_grad():
            original=model(input_ids=ids,output_hidden_states=True,use_cache=True)
            expansion.keep_last_logits(model)
            limited=model(input_ids=ids,output_hidden_states=True,use_cache=True)
        self.assertEqual(limited.logits.shape,(2,1,40))
        for a,b in zip(original.hidden_states,limited.hidden_states):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(original.past_key_values.layers,limited.past_key_values.layers):
            torch.testing.assert_close(a.keys,b.keys,rtol=0,atol=0)
            torch.testing.assert_close(a.values,b.values,rtol=0,atol=0)
        with self.assertRaises(ValueError):
            model(input_ids=ids,labels=ids)

    def test_disk_guard_and_train_only_decoder(self):
        with patch.object(expansion.shutil,'disk_usage',return_value=SimpleNamespace(free=100)):
            self.assertEqual(expansion.disk_guard(ROOT,100),100)
            with self.assertRaises(RuntimeError): expansion.disk_guard(ROOT,101)
        engine=Mock()
        heldout=[dict(source_split='validation',source_id=f'x:validation:{i}') for i in range(16)]
        with self.assertRaises(ValueError): expansion.decode_general(engine,[{}]*16,heldout,10,42)
        engine.generate.assert_not_called()



class ExpansionSupervisorTests(unittest.TestCase):
    def test_two_distinct_gpus_with_both_memory_budgets(self):
        from scripts.schedule_profile_expansion import choose_gpu_pair
        rows=[dict(index=0,uuid='A',total_mib=81559,free_mib=12000),
              dict(index=1,uuid='B',total_mib=81559,free_mib=24000),
              dict(index=2,uuid='C',total_mib=81559,free_mib=18000)]
        chosen=choose_gpu_pair(rows,.18,16)
        self.assertEqual([x['index'] for x in chosen],[2,1])
        self.assertIsNone(choose_gpu_pair(rows[:2],.18,16))

    def test_reservation_headroom_and_single_gpu_are_not_ignored(self):
        from scripts.schedule_profile_expansion import choose_gpu_pair
        rows=[dict(index=0,uuid='A',total_mib=81559,free_mib=19000),
              dict(index=1,uuid='B',total_mib=81559,free_mib=19000)]
        self.assertIsNone(choose_gpu_pair(rows,.18,16))
        self.assertIsNone(choose_gpu_pair([dict(index=0,uuid='A',total_mib=81559,free_mib=80000)],.18,16))

    def test_expanded_loader_rejects_unknown_split_before_reading(self):
        from controller.expanded_dataset import load_expanded_dataset
        with self.assertRaises(ValueError): load_expanded_dataset('/unused','../escape')
