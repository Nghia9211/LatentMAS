import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch,yaml
from controller.ppo import PPOController,sample_episodes,ppo_loss
from controller.training import logged_ppo_update
from scripts.train_expanded_controller import training_config
from scripts.train_normalized_controller import load_config

ROOT=Path(__file__).resolve().parents[1]

class NoTieAuxiliaryTests(unittest.TestCase):
    def test_only_tie_coefficient_changes_effective_training_config(self):
        plan=json.loads((ROOT/'profiles/medqa_train1024_valid128_h100_expansion_20261018_v1/plan.json').read_text())
        a=yaml.safe_load((ROOT/'configs/controller_ppo_train1024_no_warmstart32.yaml').read_text())
        b=yaml.safe_load((ROOT/'configs/controller_ppo_train1024_no_warmstart_no_tie32.yaml').read_text())
        x,y=training_config(plan,a),training_config(plan,b)
        self.assertEqual({k for k in x if x[k]!=y[k]},{'tie_weight'})
        self.assertEqual(y['tie_weight'],0);self.assertEqual(y['warmstart_epochs'],0)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'config.yaml';p.write_text(yaml.safe_dump(b))
            with self.assertRaises(ValueError):load_config(p,allow_zero_warmstart=True)
            self.assertEqual(load_config(p,allow_zero_warmstart=True,allow_tie_weight=True)['tie_weight'],0)
            for bad in [-1,.05,True,'0']:
                b['tie_weight']=bad;p.write_text(yaml.safe_dump(b))
                with self.assertRaises(ValueError):load_config(p,allow_zero_warmstart=True,allow_tie_weight=True)

    def test_zero_auxiliary_has_exact_pure_ppo_updates_without_teacher_access(self):
        torch.manual_seed(42)
        model=PPOController(9,8);control=copy.deepcopy(model)
        x=torch.randn(32,7,9);episode=sample_episodes(model,x,torch.randint(0,2,(32,8)))
        cfg=dict(optimization_epochs=4,minibatch_questions=8,clip=.2,value_weight=.5,
                 entropy_weight=.01,tie_weight=0,grad_clip=1.)
        opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0)
        reference=torch.optim.AdamW(control.parameters(),lr=.0003,weight_decay=0)
        class Forbidden:
            def __getitem__(self,key):raise AssertionError('No teacher features or tie labels allowed')
        logs=[];rng=torch.get_rng_state();calls=[]
        handle=model.register_forward_hook(lambda *args:calls.append(1))
        with patch('controller.ppo.F.cross_entropy',side_effect=AssertionError('No supervised CE')):
            loss,updates=logged_ppo_update(model,opt,x,episode,Forbidden(),Forbidden(),cfg,
                                          lambda step,row:logs.append(row))
        handle.remove();torch.set_rng_state(rng)
        for _ in range(4):
            order=torch.randperm(32)
            for start in range(0,32,8):
                idx=order[start:start+8];data={k:v[idx] for k,v in episode.items()}
                reference.zero_grad(set_to_none=True)
                expected,_=ppo_loss(control,x[idx],data,torch.ones(8,7,dtype=torch.bool),cfg)
                expected.backward();torch.nn.utils.clip_grad_norm_(control.parameters(),1.)
                reference.step()
        self.assertEqual(updates,16);self.assertEqual(len(calls),16)
        self.assertTrue(all(row['loss/tie']==0 for row in logs))
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,control.state_dict()[key],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
