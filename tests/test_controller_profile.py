"""CPU evidence for production profiling primitives; no pretrained weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
import controller.profile as cp
from latent_batch import rollout

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('profile_runner',ROOT/'scripts/profile_controller.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FeatureTests(unittest.TestCase):
    def test_known_formula_all_checkpoints(self):
        h=torch.tensor([[1+.1*t,0] for t in range(81)])
        for k in cp.GRID:
            mean=sum(.1/(1+.1*(t-1)+cp.EPS) for t in range(k-9,k+1))/10
            before=sum(.1/(1+.1*(t-1)+cp.EPS) for t in range(k-19,k-9))/10 if k>=20 else mean
            expected=torch.tensor([1,1/(1+.1*(k-10)+cp.EPS),mean,mean-before])
            torch.testing.assert_close(cp.phi4(h,k),expected,atol=1e-6,rtol=1e-5)

    def test_future_invariance_and_no_mutation(self):
        torch.manual_seed(42)
        h=torch.randn(81,16).to(torch.bfloat16)
        original=h.clone()
        for k in cp.GRID:
            future=h.clone();future[k+1:]=1000
            torch.testing.assert_close(cp.phi4(h,k),cp.phi4(future,k),atol=0,rtol=0)
            torch.testing.assert_close(cp.raw_observation(h,k),cp.raw_observation(future,k),atol=0,rtol=0)
            self.assertEqual(len(cp.raw_observation(h,k)),3*16+5)
            self.assertEqual(cp.raw_observation(h,k)[-1],k/80)
        torch.testing.assert_close(h,original,atol=0,rtol=0)

    def test_zero_and_constant_states(self):
        for h in (torch.zeros(81,3),torch.ones(81,3)):
            for k in cp.GRID:
                f=cp.phi4(h,k)
                self.assertTrue(torch.isfinite(f).all())
                self.assertTrue(f[1:].eq(0).all())

    def test_net_vs_step_change(self):
        h=torch.tensor([[1+t%2,0] for t in range(81)],dtype=torch.float32)
        self.assertEqual(cp.phi4(h,10)[1],0)
        self.assertGreater(cp.phi4(h,10)[2],0)

    def test_incomplete_invalid_nonfinite(self):
        for h,k in [(torch.ones(10,3),10),(torch.ones(81,3),15),(torch.full((81,3),float('nan')),10)]:
            with self.assertRaises(ValueError):cp.phi4(h,k)

    def test_ties_and_missing_outcomes(self):
        for q,k in [([1]*8,10),([0]*8,10),([0,1,0,1,0,0,0,1],20),([0]*7+[1],80)]:
            targets=cp.tie_targets(q)
            self.assertEqual(targets['oracle_budget'],k)
            i=10
            while i<80 and targets['checkpoint_actions'][str(i)]=='CONTINUE_10':i+=10
            self.assertEqual(i,k)
        for q in ([0]*7,[0]*7+[None]):
            with self.assertRaises(ValueError):cp.tie_targets(q)


class ObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.model=Qwen3ForCausalLM(Qwen3Config(vocab_size=40,hidden_size=32,
            intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
            num_key_value_heads=2,head_dim=8,eos_token_id=None,pad_token_id=0)).eval()
        cls.wrapper=SimpleNamespace(HF_model=cls.model,_apply_latent_realignment=lambda h,_:h)

    def test_observer_on_off_prefix_cache_and_nonmutation(self):
        ids=torch.tensor([[0,4,5],[6,7,8]])
        mask=ids.ne(0)
        cache0,emb0=rollout(self.wrapper,ids,mask,latent_steps=3)
        seen=[]
        def observer(**record):
            seen.append((record['step'],record['hidden_last_position'].clone(),
                         None if record['latent_input'] is None else record['latent_input'].clone()))
            record['hidden_last_position'].zero_()
            if record['latent_input'] is not None:record['latent_input'].zero_()
        cache1,emb1=rollout(self.wrapper,ids,mask,latent_steps=3,observer=observer)
        self.assertEqual([r[0] for r in seen],[0,1,2,3])
        self.assertEqual(seen[0][1].shape,(2,3,32))
        self.assertIsNone(seen[0][2])
        for a,b in zip(emb0,emb1):torch.testing.assert_close(a,b,atol=0,rtol=0)
        for a,b in zip(cache0.layers,cache1.layers):
            torch.testing.assert_close(a.keys,b.keys,atol=0,rtol=0)
            torch.testing.assert_close(a.values,b.values,atol=0,rtol=0)
        _,single=rollout(self.wrapper,torch.tensor([[4,5]]),latent_steps=0,
            observer=lambda **r: seen.append(('single',r['hidden_last_position'],None)))
        torch.testing.assert_close(seen[0][1][0],seen[-1][1][0],atol=3e-6,rtol=3e-5)

    def test_collector_all_step_shape_actual_inputs_and_prefix_reuse(self):
        collector=cp.TrajectoryCollector()
        ids=torch.tensor([[4,5]])
        _,emb=rollout(self.wrapper,ids,latent_steps=80,observer=collector)
        tensors=collector.tensors()
        self.assertEqual(tensors['hidden_last_position_all_layers'].shape,(81,3,32))
        self.assertEqual(tensors['latent_inputs_actual'].shape,(80,32))
        torch.testing.assert_close(tensors['latent_inputs_actual'],emb[0][2:],atol=0,rtol=0)
        prompts={'A_prompt_embeddings_actual':emb[0][:2]}
        for k in (10,20,40,80):
            _,independent=rollout(self.wrapper,ids,latent_steps=k)
            torch.testing.assert_close(cp.reconstruct_prefix(prompts,tensors,k),independent[0],atol=0,rtol=0)

    def test_collector_rejects_missing_order_batch(self):
        collector=cp.TrajectoryCollector()
        with self.assertRaises(ValueError):collector.tensors()
        for step,shape in [(1,(1,3,32)),(0,(2,3,32))]:
            with self.assertRaises(ValueError):collector(step=step,hidden_last_position=torch.ones(shape),latent_input=None)

    def test_observer_cannot_stop_rollout(self):
        with self.assertRaisesRegex(ValueError,'return None'):
            rollout(self.wrapper,torch.tensor([[4,5]]),latent_steps=1,observer=lambda **r:True)


class DataAndArchiveTests(unittest.TestCase):
    def fixture(self,root):
        rows=[dict(source_id=f'train:{i}',source_split='train',source_row=i,
            question=q,options={'A':'one','B':'two','C':'three','D':'four'},
            answer_idx='B',answer='two',question_sha256='fixture')
              for i,q in enumerate(['Q1','Q2','Q1'])]
        cp.write_jsonl(root/'train.jsonl',rows)
        cp.write_json(root/'manifest.json',{'splits':{'train':{'rows':3,'file':'train.jsonl',
            'sha256':cp.file_hash(root/'train.jsonl')}}})

    def test_official_schema_options_once_dedup_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            items,_=cp.load_official_medqa(root)
            self.assertEqual(items[0]['question'],'Q1\n\nA. one\nB. two\nC. three\nD. four')
            self.assertEqual(items[0]['gold'],'b')
            self.assertEqual(cp.make_subset(items,2,42),cp.make_subset(items,2,42))
            self.assertEqual(len({r['question_stem'] for r in cp.make_subset(items,2)}),2)
            with self.assertRaises(ValueError):cp.make_subset(items,3)

    def test_dataset_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            with (root/'train.jsonl').open('a') as handle:handle.write('\n')
            with self.assertRaisesRegex(ValueError,'checksum'):cp.load_official_medqa(root)

    def test_archive_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cp.write_json(root/'one.json',{'a':1})
            with self.assertRaises(FileExistsError):cp.write_json(root/'one.json',{'a':2})
            self.assertEqual(json.loads((root/'one.json').read_text()),{'a':1})
            cp.write_tensors(root/'one.safetensors',{'a':torch.ones(2,dtype=torch.bfloat16)})
            with self.assertRaises(FileExistsError):cp.write_tensors(root/'one.safetensors',{'a':torch.zeros(2)})
            self.assertFalse(list(root.glob('.pending-*')))

    def test_checksum_and_missing_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.assertEqual(cp.archive_coverage(root,['missing'])['missing'],1)
            cp.write_json(root/'x.json',{'a':1})
            cp.verify_file_manifest(root,{'x.json':cp.file_hash(root/'x.json')})
            with self.assertRaises(ValueError):cp.verify_file_manifest(root,{'x.json':'bad'})

    def test_protocol_and_feature_allowlist(self):
        config=runner.load_config(ROOT/'configs/profile_medqa_pilot.yaml')
        self.assertEqual(config['max_new_tokens'],4096)
        self.assertEqual(config['pilot_samples']*len(config['checkpoints']),256)
        schema=runner.feature_schema()
        self.assertEqual(schema['phi_names'],cp.PHI_NAMES)
        self.assertEqual(schema['policy_metadata'],['k/80'])
        self.assertFalse(schema['availability_masks'])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.yaml'
            import yaml
            config['max_new_tokens']=256
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError,'max_new_tokens'):runner.load_config(path)

    def test_pinned_revision_and_dtype_plumbing(self):
        import models
        args=SimpleNamespace(method='latent_mas',device2='cpu',seed=42,
            model_revision='a'*40,vllm_dtype='bfloat16',enable_prefix_caching=True)
        with patch.object(models,'_HAS_VLLM',True),patch.object(models,'LLM') as engine, \
             patch.object(models,'AutoTokenizer'),patch.object(models,'AutoModelForCausalLM'), \
             patch.object(models.ModelWrapper,'_ensure_latent_realign_matrix'):
            models.ModelWrapper('fake','cpu',use_vllm=True,args=args)
        self.assertEqual(engine.call_args.kwargs['revision'],'a'*40)
        self.assertEqual(engine.call_args.kwargs['dtype'],'bfloat16')
        self.assertEqual(engine.call_args.kwargs['tokenizer_revision'],'a'*40)


class PipelineTests(unittest.TestCase):
    def test_capture_parity_archive_and_resume_with_tiny_bf16_model(self):
        from models import ModelWrapper
        class Tokenizer:
            chat_template='fixture'
            pad_token_id=0
            def apply_chat_template(self,messages,**kwargs):
                return '<|im_start|>user\n'+messages[-1]['content']+'\nassistant\n'
            def __call__(self,texts,**kwargs):
                if isinstance(texts,str):return {'input_ids':[3+ord(c)%30 for c in texts]}
                rows=[[3+ord(c)%30 for c in text] for text in texts]
                width=max(map(len,rows))
                return {'input_ids':torch.tensor([r+[0]*(width-len(r)) for r in rows]),
                    'attention_mask':torch.tensor([[1]*len(r)+[0]*(width-len(r)) for r in rows])}
            def convert_ids_to_tokens(self,ids):return list(map(str,ids))
        class Engine:
            def reset_prefix_cache(self):return True
            def generate(self,inputs,params,**kwargs):
                return [SimpleNamespace(outputs=[SimpleNamespace(text=r'\boxed{B}',
                    token_ids=[7,8,9],finish_reason='stop',stop_reason=None)]) for _ in inputs]
        torch.manual_seed(42)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=40,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            eos_token_id=None,pad_token_id=0)).to(torch.bfloat16).eval()
        w=ModelWrapper.__new__(ModelWrapper)
        w.model=w.HF_model=model
        w.embedding_layer=model.get_input_embeddings()
        w.device=torch.device('cpu');w.HF_device='cpu'
        w.tokenizer=Tokenizer();w.vllm_engine=Engine()
        w._apply_latent_realignment=lambda h,_:h
        config=runner.load_config(ROOT/'configs/profile_medqa_pilot.yaml')
        config.update(device='cpu',device2='cpu',model_name='Qwen/tiny',parity_samples=1,
                      parity_checkpoints=[10,20])
        args=SimpleNamespace(**config)
        item=dict(source_id='fixture:train:0',source_split='train',question='q',gold='b',solution='b')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            captures=runner.parity(w,args,[item],root)
            report=json.loads((root/'parity_report.json').read_text())
            self.assertTrue(report['passed'])
            runner.capture_sample(w,args,item,root,captures[item['source_id']])
            coverage=cp.archive_coverage(root,[item['source_id']])
            self.assertTrue(coverage['all_complete'])
            self.assertEqual(coverage['completed_b_completions'],8)
            marker=root/'trajectories'/item['source_id']/'complete.json'
            original=cp.file_hash(marker)
            runner.capture_sample(w,args,item,root)
            self.assertEqual(cp.file_hash(marker),original)
            self.assertEqual(runner.parity(w,args,[item],root),{})


if __name__=='__main__':unittest.main()
