import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch,yaml
from controller.normalized import NormalizedPPOController,NormalizedWarmStartController
from scripts.train_normalized_controller import warmstart,load_config,stopping_probabilities
from scripts.train_expanded_controller import training_config

ROOT=Path(__file__).resolve().parents[1]

class NoWarmstartTests(unittest.TestCase):
    def test_zero_epochs_explicitly_opt_in_only_and_negative_rejected(self):
        settings=yaml.safe_load((ROOT/'configs/controller_ppo_train1024_baseline32.yaml').read_text())
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'cfg.yaml';settings['warmstart_epochs']=0;p.write_text(yaml.safe_dump(settings))
            with self.assertRaises(ValueError):load_config(p)
            self.assertEqual(load_config(p,allow_zero_warmstart=True)['warmstart_epochs'],0)
            for key,value in [('warmstart_epochs',-1),('iterations',0),('validation_every',0)]:
                s=dict(settings);s[key]=value;p.write_text(yaml.safe_dump(s))
                with self.assertRaises(ValueError):load_config(p,allow_zero_warmstart=True)

    def test_cold_init_uses_no_labels_no_warm_update_and_exact_random_actor(self):
        torch.manual_seed(42)
        x=torch.randn(8,7,7685);x[...,-1]=torch.arange(1,8)/8
        model=NormalizedPPOController(7685,4);model.normalizer.fit(x,source_split='train')
        opt=torch.optim.AdamW(model.parameters(),lr=.0003)
        cfg=dict(warmstart_epochs=0,width=4,device='cpu',normalization_eps=1e-5,projection_norm=True)
        captured={}
        def factory(*args,**kwargs):
            proto=NormalizedWarmStartController(*args,**kwargs)
            captured['encoder']=copy.deepcopy(proto.encoder.state_dict())
            captured['actor']=copy.deepcopy(proto.actor.state_dict())
            return proto
        class PoisonLabels:
            def __getitem__(self,key):raise AssertionError('Cold init may not read profile labels or features')
        class Tracker:
            def __init__(self):self.logs=[]
            def log(self,row):self.logs.append(row)
        tracker=Tracker()
        with tempfile.TemporaryDirectory() as td:
            with patch('scripts.train_normalized_controller.train_update',side_effect=AssertionError('No warm optimizer step')):
                with patch('scripts.train_normalized_controller.NormalizedWarmStartController',side_effect=factory):
                    checkpoint=warmstart(model,opt,PoisonLabels(),cfg,Path(td),tracker)
            for name in ('encoder','actor'):
                for k,v in getattr(model,name).state_dict().items():torch.testing.assert_close(v,captured[name][k],rtol=0,atol=0)
            self.assertTrue(torch.equal(model.critic.weight,torch.zeros_like(model.critic.weight)))
            self.assertTrue(torch.equal(model.critic.bias,torch.full_like(model.critic.bias,.5)))
            self.assertFalse(opt.state)
            result=json.loads((checkpoint.parent/'result.json').read_text())
            self.assertTrue(result['warmstart_skipped']);self.assertEqual(result['optimizer_updates'],0)
            self.assertFalse(result['labels_used']);self.assertFalse((Path(td)/'warmstart').exists())
            with patch('scripts.train_normalized_controller.NormalizedWarmStartController',side_effect=AssertionError('Do not reinitialize committed checkpoint')):
                self.assertEqual(warmstart(model,opt,PoisonLabels(),cfg,Path(td),tracker),checkpoint)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            self.assertEqual(saved['iteration'],0);self.assertFalse(saved['optimizer']['state'])

    def test_uniform_binary_actions_do_not_imply_uniform_budgets(self):
        p=stopping_probabilities(torch.zeros(3,7,2))
        expected=torch.tensor([.5,.25,.125,.0625,.03125,.015625,.0078125,.0078125])
        torch.testing.assert_close(p,expected.expand(3,8),rtol=0,atol=0)
        torch.testing.assert_close(p.sum(1),torch.ones(3),rtol=0,atol=0)
        self.assertAlmostEqual(float((p[0]*torch.arange(10,81,10)).sum()),19.921875)
        with self.assertRaises(ValueError):stopping_probabilities(torch.zeros(3,8,2))

    def test_baseline_and_ablation_only_change_warmstart_training_config(self):
        root=ROOT/'profiles/medqa_train1024_valid128_h100_expansion_20261018_v1'
        plan=json.loads((root/'plan.json').read_text())
        a=yaml.safe_load((ROOT/'configs/controller_ppo_train1024_baseline32.yaml').read_text())
        b=yaml.safe_load((ROOT/'configs/controller_ppo_train1024_no_warmstart32.yaml').read_text())
        x,y=training_config(plan,a),training_config(plan,b)
        self.assertEqual({k for k in x if x[k]!=y[k]},{'warmstart_epochs'})
        self.assertEqual(y['warmstart_epochs'],0);self.assertEqual(y['tie_weight'],.1)

if __name__=='__main__':unittest.main()
