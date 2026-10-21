import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from controller.profile import raw_observation
from controller.ppo import PPOController,sample_episodes,ppo_loss,greedy_budgets,OnlineStopper
from controller.warmstart import WarmStartController
from controller.training import replay_iteration as iteration
from controller.matched import select_test16


CONFIG=dict(clip=.2,value_weight=.5,entropy_weight=.01,tie_weight=.1,
    optimization_epochs=2,minibatch_questions=2,grad_clip=1.)


def constant_policy(continue_action=False):
    model=PPOController(17,8)
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.copy_(torch.tensor([-100.,100.] if continue_action else [100.,-100.]))
    return model


class PPOTests(unittest.TestCase):
    def test_first_stop_and_only_visited_forwarded(self):
        model=constant_policy();calls=[]
        model.register_forward_hook(lambda m,a,o:calls.append(a[0].shape))
        x=torch.randn(3,7,17);q=torch.tensor([[1,0,0,0,0,0,0,0]]*3)
        e=sample_episodes(model,x,q)
        self.assertEqual(calls,[torch.Size([3,17])])
        self.assertEqual(e['visited'].sum().item(),3)
        self.assertEqual(e['stop_indices'].tolist(),[0]*3)
        self.assertEqual(e['returns'][:,0].tolist(),[1.]*3)
        self.assertFalse(e['old_logprobs'].requires_grad)

    def test_cap_has_no_actor_action_and_terminal_return(self):
        model=constant_policy(True);calls=[]
        model.register_forward_hook(lambda m,a,o:calls.append(a[0].shape))
        e=sample_episodes(model,torch.randn(2,7,17),torch.tensor([[0]*7+[1],[1]*7+[0]]))
        self.assertEqual(len(calls),7)
        self.assertEqual(e['visited'].sum().item(),14)
        self.assertEqual(e['stop_indices'].tolist(),[7,7])
        self.assertEqual(e['rewards'].tolist(),[1.,0.])
        torch.testing.assert_close(e['returns'][0],torch.ones(7))

    def test_old_logprobs_and_advantages_remain_fixed(self):
        model=PPOController(17,8)
        x=torch.randn(4,7,17);q=torch.ones(4,8)
        e=sample_episodes(model,x,q)
        old=e['old_logprobs'].clone();advantages=e['advantages'].clone()
        loss,parts=ppo_loss(model,x,e,torch.zeros(4,7,dtype=torch.bool),CONFIG)
        self.assertAlmostEqual(float(parts['approx_kl']),0,places=6)
        self.assertEqual(float(parts['clip_fraction']),0)
        loss.backward()
        torch.testing.assert_close(e['old_logprobs'],old,rtol=0,atol=0)
        torch.testing.assert_close(e['advantages'],advantages,rtol=0,atol=0)

    def test_clipping_and_episode_weighting(self):
        model=PPOController(17,8)
        with torch.no_grad():
            model.actor.weight.zero_();model.actor.bias.copy_(torch.tensor([.9,.1]).log())
        x=torch.zeros(2,7,17)
        mask=torch.tensor([[1,0,0,0,0,0,0],[1,1,1,0,0,0,0]],dtype=torch.bool)
        e=dict(actions=torch.zeros(2,7,dtype=torch.long),old_logprobs=torch.full((2,7),torch.tensor(.5).log()),
            visited=mask,advantages=mask.float(),returns=torch.ones(2,7))
        cfg=dict(CONFIG,value_weight=0,entropy_weight=0,tie_weight=0)
        loss,parts=ppo_loss(model,x,e,torch.zeros_like(mask),cfg)
        self.assertAlmostEqual(float(parts['actor'].detach()),-1.2*(1+3)/2,places=5)
        self.assertAlmostEqual(float(loss.detach()),-2.4,places=5)

    def test_warmstart_not_hindsight_critic(self):
        warm=WarmStartController('raw_mlp',17,8)
        model=PPOController(17,8);model.load_warmstart(warm)
        torch.testing.assert_close(model.actor.weight,warm.actor.weight)
        self.assertEqual(model.critic.weight.count_nonzero().item(),0)
        self.assertEqual(float(model.critic.bias.detach()),.5)

    def test_real_update_finite_and_changes_weights(self):
        model=PPOController(17,8);before=copy.deepcopy(model.state_dict())
        x=torch.randn(4,7,17);q=torch.tensor([[1,0,1,0,1,0,1,0]]*4).float()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        e,row=iteration(model,optimizer,x,q,torch.ones(4,7,dtype=torch.bool),CONFIG)
        self.assertEqual(row['updates'],4)
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.state_dict().items()))
        self.assertTrue(all(torch.isfinite(v).all() for v in model.state_dict().values()))

    def test_online_replay_greedy_parity_and_cap(self):
        hidden=torch.randn(81,4).bfloat16()
        x=torch.stack([raw_observation(hidden,k) for k in range(10,80,10)])[None]
        for model in (PPOController(17,8),constant_policy(True),constant_policy(False)):
            expected=int(greedy_budgets(model,x)[0]);stopper=OnlineStopper(model)
            stopper.begin(hidden[0])
            for k in range(1,81):
                if stopper(step=k,hidden_last_position=hidden[k][None]):break
            self.assertEqual(k,expected)
            self.assertEqual(len(stopper.hidden),k+1)
            self.assertTrue(all(d.get('hidden_coverage_end',80)==d['k'] for d in stopper.decisions))
            self.assertEqual(stopper.decisions[-1]['forced'],k==80)

    def test_online_h0_and_order_required(self):
        stopper=OnlineStopper(PPOController(17,8))
        with self.assertRaises(ValueError):stopper(step=1,hidden_last_position=torch.randn(1,4))
        stopper.begin(torch.randn(4))
        with self.assertRaises(ValueError):stopper(step=2,hidden_last_position=torch.randn(1,4))

    def test_historical_test16_matches_official_not_train(self):
        root=Path(__file__).resolve().parents[1]
        items=select_test16(root/'data/medqa_usmle_4options')
        self.assertEqual(len(items),16)
        self.assertEqual([i['historical_index'] for i in items],list(range(16)))
        self.assertTrue(all(i['source_split']=='test' for i in items))

    def test_rollout_actually_stops_and_matches_fixed_prefix(self):
        from latent_batch import rollout
        from transformers import Qwen3Config,Qwen3ForCausalLM
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=40,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)).eval()
        wrapper=SimpleNamespace(HF_model=model,_apply_latent_realignment=lambda h,_:h)
        ids=torch.tensor([[3,4,5]]);seen=[]
        stop=lambda **kw:kw['step']==3
        observer=lambda **kw:seen.append(kw['step'])
        past,adaptive=rollout(wrapper,ids,latent_steps=8,observer=observer,stopping_policy=stop)
        _,fixed=rollout(wrapper,ids,latent_steps=3)
        torch.testing.assert_close(adaptive[0],fixed[0],atol=0,rtol=0)
        self.assertEqual(seen,[0,1,2,3])
        self.assertEqual(past._latent_next_position.tolist(),[6])
        self.assertEqual(past._latent_attention_mask.shape,(1,6))
        with self.assertRaises(ValueError):rollout(wrapper,ids.expand(2,3),latent_steps=3,stopping_policy=stop)


if __name__=='__main__':unittest.main()
