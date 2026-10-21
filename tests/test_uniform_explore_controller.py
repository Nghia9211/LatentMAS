import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.distributions import Categorical
from controller.normalized import NormalizedWarmStartController, NormalizedPPOController
from controller.uniform_explore import (UniformBudgetPPOController, budget_probabilities,
    uniform_log_prior, exact_visited_kl, tie_weight, load_uniform_checkpoint)
from controller.ppo import sample_episodes, ppo_loss
from controller.warmstart import supervision


class UniformExploreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x=torch.randn(24,7,17)
        self.x[:,:,-1]=torch.arange(1,8)/8
        self.model=UniformBudgetPPOController(17,12)
        self.model.normalizer.fit(self.x,source_split='train')

    def test_uniform_budget_not_uniform_binary_actions(self):
        logits,_=self.model(self.x)
        expected=1/torch.arange(8,1,-1).float()
        torch.testing.assert_close(logits.softmax(-1)[0,:,0],expected)
        torch.testing.assert_close(budget_probabilities(self.model,self.x),torch.full((24,8),.125))
        self.assertFalse(torch.allclose(logits.softmax(-1)[0,:,0],torch.full((7,),.5)))

    def test_prior_only_uses_current_step_and_supports_masked_padding(self):
        changed=self.x.clone();changed[:,:,:-1]*=100
        torch.testing.assert_close(uniform_log_prior(self.x),uniform_log_prior(changed))
        self.assertTrue(torch.isfinite(uniform_log_prior(torch.zeros(2,17))).all())
        for value in [1.,.13,-.125]:
            bad=self.x.clone();bad[0,0,-1]=value
            with self.assertRaises(ValueError):uniform_log_prior(bad)

    def test_residual_is_learnable_not_forced_uniform(self):
        initial=budget_probabilities(self.model,self.x)
        with torch.no_grad():self.model.actor.bias[0]=2.
        changed=budget_probabilities(self.model,self.x)
        self.assertTrue((changed[:,0]>initial[:,0]).all())
        loss=-self.model(self.x)[0].log_softmax(-1)[:,:,1].mean()
        loss.backward()
        self.assertGreater(float(self.model.actor.weight.grad.norm()),0)

    def test_warmstart_retains_encoder_resets_stop_actor_and_fresh_critic(self):
        warm=NormalizedWarmStartController(17,12)
        warm.normalizer.fit(self.x,source_split='train')
        with torch.no_grad():warm.actor.bias[:]=torch.tensor([100.,-100.])
        self.model.load_warmstart(warm)
        torch.testing.assert_close(self.model.encoder[0].weight,warm.encoder[0].weight)
        self.assertEqual(float(self.model.actor.weight.detach().abs().max()),0)
        self.assertEqual(float(self.model.critic.weight.detach().abs().max()),0)
        self.assertEqual(float(self.model.critic.bias.detach()[0]),.5)
        torch.testing.assert_close(budget_probabilities(self.model,self.x),torch.full((24,8),.125))

    def test_old_and_new_ppo_probabilities_use_same_prior(self):
        q=torch.randint(0,2,(24,8)).float()
        ep=sample_episodes(self.model,self.x,q)
        logits,_=self.model(self.x)
        logp=Categorical(logits=logits).log_prob(ep['actions'])
        torch.testing.assert_close(logp[ep['visited']],ep['old_logprobs'][ep['visited']])
        self.assertAlmostEqual(exact_visited_kl(self.model,self.x,logits.detach(),ep['visited']),0.)
        cfg=dict(clip=.2,value_weight=.5,entropy_weight=.01,tie_weight=0.)
        loss,_=ppo_loss(self.model,self.x,ep,torch.zeros(24,7,dtype=torch.bool),cfg)
        self.assertTrue(torch.isfinite(loss));loss.backward()

    def test_all_wrong_kept_no_fabricated_high_budget_reward(self):
        q=torch.zeros(24,8)
        targets,_=supervision(q[0])
        self.assertTrue((targets==0).all())  # Historical tie labels unchanged.
        ep=sample_episodes(self.model,self.x,q)
        self.assertEqual(len(ep['rewards']),24)
        self.assertTrue((ep['rewards']==0).all())
        torch.testing.assert_close(budget_probabilities(self.model,self.x),torch.full((24,8),.125))

    def test_tie_schedule(self):
        expected=[0,0,0,0,.025,.05,.075,.1,.1]
        for n,v in enumerate(expected,1):self.assertAlmostEqual(tie_weight(n),v)
        with self.assertRaises(ValueError):tie_weight(0)

    def test_checkpoint_versions_and_loader(self):
        plain=NormalizedPPOController(17,12)
        plain.normalizer.fit(self.x,source_split='train')
        with self.assertRaises(ValueError):self.model.load_state_dict(plain.state_dict(),strict=False)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint.pt'
            torch.save(dict(model=self.model.state_dict()),p)
            loaded=load_uniform_checkpoint(p)
            torch.testing.assert_close(loaded(self.x)[0],self.model(self.x)[0])






if __name__=='__main__':unittest.main()
