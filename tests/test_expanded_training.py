import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from controller.normalized import NormalizedPPOController
from controller.sampler import ShuffledQuestionSampler
from scripts.train_normalized_controller import warmstart,train
from scripts.train_expanded_controller import approved_items,training_config,schedule

class ExpandedTrainingTests(unittest.TestCase):
    def test_full_epoch_has_exactly1024_unique_questions_and_no_skip(self):
        ids=['m:train:'+str(i) for i in range(1024)]
        sampler=ShuffledQuestionSampler(ids,42)
        seen=[i for n in range(1,33) for i in sampler.plan(n)['source_ids']]
        self.assertEqual(len(seen),1024);self.assertEqual(set(seen),set(ids))
        self.assertEqual(sampler.plan(32)['epoch'],1)
        self.assertEqual(sampler.plan(33)['epoch'],2)

    def test_baseline_preserves_old_ppo_math_and_device(self):
        cfg=dict(width=128,optimization_epochs=4,minibatch_questions=8,time_penalty=0,
            tie_weight=.1,entropy_weight=.01,learning_rate=.0003,temperature=.6,generate_bs=16,
            device='cuda:0',device2='cuda:1')
        settings=dict(iterations=32,warmstart_epochs=50,projection_norm=True,eps=1e-5)
        out=training_config(dict(config=cfg),settings)
        for k,v in cfg.items():
            if k!='device':self.assertEqual(out[k],v)
        self.assertEqual(out['device'],'cuda:1');self.assertEqual(out['train_samples'],1024)
        cfg['tie_weight']=0
        with self.assertRaises(ValueError):training_config(dict(config=cfg),settings)

    def test_validation_schedule_includes_warmstart_and_final(self):
        self.assertEqual(schedule(32,8),[0,8,16,24,32])
        self.assertEqual(schedule(10,8),[0,8,10])

    def test_exact_official_id_order_and_split_firewall(self):
        ids=['m:train:'+str(i) for i in range(1024)]
        pool=[dict(source_id=i,source_split='train',question_stem=i,question=i+'\n\nA. a\nB. b\nC. c\nD. d') for i in ids]
        plan=dict(selection=dict(source_ids=ids,dataset_sha256='abc'),config=dict(data_dir='unused'))
        with patch('scripts.train_expanded_controller.load_official_medqa',return_value=(pool,dict(sha256='abc'))):
            items=approved_items(plan,{},'train')
            self.assertEqual([x['source_id'] for x in items],ids)
            changed=copy.deepcopy(plan);changed['selection']['dataset_sha256']='changed'
            with self.assertRaises(ValueError):approved_items(changed,{},'train')
            bad=copy.deepcopy(pool);bad[0]['source_split']='validation'
            with patch('scripts.train_expanded_controller.load_official_medqa',return_value=(bad,dict(sha256='abc'))):
                with self.assertRaises(ValueError):approved_items(plan,{},'train')
        with self.assertRaises(ValueError):approved_items(plan,{},'test')

    def test_override_identity_mismatch_rejected_before_gpu_or_tracker(self):
        m=dict(config=dict(train_samples=1024),train_ids=['m:train:'+str(i) for i in range(1024)],validation_ids=['v:validation:0'])
        with self.assertRaisesRegex(ValueError,'identities differ'):
            train(Path('unused'),m,dict(features=torch.zeros(1024,7,1)),None,None,
                train_items=[],validation_items=[])
        with self.assertRaisesRegex(ValueError,'both declared'):
            train(Path('unused'),m,{},None,None,train_items=[])

    def test_actual_warmstart_full1024_updates_all_and_logs_correct_pool(self):
        torch.manual_seed(7)
        x=torch.randn(1024,7,7685);x[...,-1]=torch.arange(1,8)/8
        model=NormalizedPPOController(7685,4);model.normalizer.fit(x,source_split='train')
        opt=torch.optim.AdamW(model.parameters(),lr=.0003)
        d=dict(features=x,actions=torch.randint(0,2,(1024,7)),q_targets=torch.rand(1024,7,2))
        cfg=dict(width=4,device='cpu',normalization_eps=1e-5,projection_norm=True,
                 warmstart_epochs=1,learning_rate=.0003,grad_clip=1.)
        class Tracker:
            def log(self,row):pass
        with tempfile.TemporaryDirectory() as td:
            path=warmstart(model,opt,d,cfg,Path(td),Tracker())
            result=json.loads((path.parent/'result.json').read_text())
            self.assertEqual(result['train_questions'],1024)
            self.assertEqual(result['optimizer_updates'],1)
            self.assertFalse(result['validation_used_for_gradient'])
            self.assertFalse(opt.state)
            saved=torch.load(path,map_location='cpu',weights_only=True)
            self.assertEqual(saved['iteration'],0)
            self.assertTrue(torch.equal(saved['model']['normalizer.phi_mean'],model.normalizer.phi_mean))

if __name__=='__main__':unittest.main()
