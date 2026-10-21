import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from controller.server_training import repeat_targets, current_sources
from controller.checkpoints import load_controller_checkpoint
from controller.normalized import NormalizedPPOController, SCHEMA as NORMALIZED
from controller.uniform_explore import UniformBudgetPPOController, SCHEMA as UNIFORM
from controller.ppo import PPOController


def fixture():
    selections={s:dict(source_ids=[f'medqa:{s}:1']) for s in ('train','validation')}
    rows=[]
    for s in selections:
        for k in range(10,81,10):
            v=([1,0,0] if k==10 else [1,1,0]) if s=='train' else [0,0,0]
            rows.append(dict(source_id=f'medqa:{s}:1',source_split=s,k=k,seeds=[42,43,44],
                correctness=v,correct_count=sum(v),trials=3,success_fraction=sum(v)/3))
    return rows,selections


class ServerTrainingTests(unittest.TestCase):
    def test_fraction_targets_order_and_tie(self):
        rows,sel=fixture()
        original=copy.deepcopy(rows)
        d=repeat_targets(list(reversed(rows)),sel)
        self.assertEqual(rows,original)
        self.assertAlmostEqual(float(d['train']['q_targets'][0,0,0]),1/3,places=6)
        self.assertAlmostEqual(float(d['train']['q_targets'][0,0,1]),2/3,places=6)
        self.assertEqual(d['train']['actions'][0].tolist(),[1,0,0,0,0,0,0])
        self.assertEqual(set(d['validation']),{'correctness'})

    def test_validation_labels_cannot_change_train_targets(self):
        rows,sel=fixture()
        before=repeat_targets(rows,sel)['train']
        for r in rows:
            if r['source_split']=='validation':
                r.update(correctness=[1,1,1],correct_count=3,success_fraction=1.)
        after=repeat_targets(rows,sel)['train']
        for k in before: torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)

    def test_incomplete_duplicate_wrong_split_or_fraction_rejected(self):
        rows,sel=fixture()
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError): repeat_targets(bad,sel)
        for key,value in [('source_split','test'),('success_fraction',1.),('seeds',[42,43,45])]:
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError): repeat_targets(bad,sel)

    def test_new_source_freeze_excludes_old_moved_files_and_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'repo';root.mkdir()
            for s in ('controller','scripts','methods','tests'): (root/s).mkdir()
            (root/'controller/new.py').write_text('new')
            (root/'config.yaml').write_text('schema: test')
            hashes=current_sources(root,root/'config.yaml')
            self.assertIn('controller/new.py',hashes)
            self.assertNotIn('controller_old.py',hashes)
            external=Path(d)/'external.py';external.write_text('outside')
            (root/'controller/escape.py').symlink_to(external)
            with self.assertRaises(ValueError): current_sources(root,root/'config.yaml')

    def test_load_raw_normalized_uniform_and_preserve_statistics(self):
        x=torch.randn(3,7,17);x[...,-1]=torch.arange(1,8)/8
        for cls,schema in [(PPOController,'raw'),(NormalizedPPOController,NORMALIZED),
                           (UniformBudgetPPOController,UNIFORM)]:
            model=cls(17,8)
            if schema!='raw': model.normalizer.fit(x,source_split='train')
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'checkpoint.pt';torch.save(dict(model=model.state_dict(),iteration=8),path)
                loaded=load_controller_checkpoint(path,expected_schema=schema)
                with torch.no_grad():
                    a,b=model(x);c,e=loaded(x)
                torch.testing.assert_close(a,c,rtol=0,atol=0)
                torch.testing.assert_close(b,e,rtol=0,atol=0)
                with self.assertRaises(ValueError):
                    load_controller_checkpoint(path,expected_schema='wrong')

    def test_corrupt_normalizer_not_silently_reinterpreted_as_raw(self):
        model=PPOController(17,8)
        state=model.state_dict();state['normalizer.fitted']=torch.tensor(True)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.pt';torch.save(dict(model=state),p)
            with self.assertRaises(ValueError): load_controller_checkpoint(p)



class RuntimeMemoryTests(unittest.TestCase):
    def test_hf_reservation_is_held_through_compile_and_reused_before_device_move(self):
        import weakref
        from types import SimpleNamespace
        from unittest.mock import patch, Mock
        import models
        references=[]; events=[]
        class Reservation: pass
        def allocate(*a,**kw):
            value=Reservation();references.append(weakref.ref(value));events.append('reserve')
            return value
        def engine(**kw):
            self.assertIsNotNone(references[0]());events.append('engine')
            self.assertEqual(kw['max_model_len'],8192)
            self.assertEqual(kw['max_num_seqs'],16)
            return Mock()
        hf=Mock();hf.eval.return_value=hf
        def load_hf(*a,**kw):
            self.assertIsNotNone(references[0]());events.append('load_cpu');return hf
        def move(device):
            self.assertIsNone(references[0]());events.append('move_to_gpu');return hf
        hf.to.side_effect=move
        tokenizer=Mock();tokenizer.pad_token_id=0
        args=SimpleNamespace(method='latent_mas',device2='cpu',hf_memory_reservation_gib=.01,
            vllm_max_model_len=8192,vllm_max_num_seqs=16,vllm_max_num_batched_tokens=4096)
        with patch.object(models,'_HAS_VLLM',True),patch.object(models,'LLM',side_effect=engine,create=True), \
             patch.object(models.torch,'empty',side_effect=allocate), \
             patch.object(models.AutoModelForCausalLM,'from_pretrained',side_effect=load_hf), \
             patch.object(models.AutoTokenizer,'from_pretrained',return_value=tokenizer), \
             patch.object(models.ModelWrapper,'_ensure_latent_realign_matrix',return_value=None):
            models.ModelWrapper('mock-model','cpu',True,args)
        self.assertEqual(events,['reserve','engine','load_cpu','move_to_gpu'])

    def test_invalid_runtime_limits_fail_before_engine_or_model_loading(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import models
        with patch.object(models,'_HAS_VLLM',True),patch.object(models,'LLM',create=True) as llm:
            for value in (-1,0,False,1.5):
                with self.assertRaises(ValueError):
                    models.ModelWrapper('mock','cpu',True,SimpleNamespace(vllm_max_model_len=value))
            for value in (-1,True,33):
                with self.assertRaises(ValueError):
                    models.ModelWrapper('mock','cpu',True,SimpleNamespace(hf_memory_reservation_gib=value))
            llm.assert_not_called()
