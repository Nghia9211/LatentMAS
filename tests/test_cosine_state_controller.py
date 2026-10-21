import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from controller import cosine_state as s
from controller.normalized import ObservationNormalizer,NormalizedPPOController
from controller.profile import raw_observation as old_observation,write_jsonl
from controller.sampler import ShuffledQuestionSampler
from controller.training import logged_ppo_update
from scripts import run_pooled_controller_ppo as pooled
from scripts import train_controller_from_profile as val
from scripts.train_cosine_state_ppo import protocol
CFG=dict(optimization_epochs=4,minibatch_questions=8,clip=.2,value_weight=.5,entropy_weight=.01,tie_weight=0,grad_clip=1.)
def make_model():
 x=torch.randn(32,7,2562);model=s.CosinePPOController()
 model.normalizer.fit(x,source_split="train")
 return model
class CosineStateTests(unittest.TestCase):
 def test_features_match_checkpoint10_cosine_and_ignore_future(self):
  h=torch.randn(81,2560).bfloat16()
  full=torch.stack([old_observation(h,k) for k in range(10,71,10)])
  selected=s.select_profile_features(full[None])[0]
  for t,k in enumerate(range(10,71,10)):
   live=s.raw_observation(h,k)
   torch.testing.assert_close(live,selected[t],rtol=1e-6,atol=1e-6)
   expected=1-torch.nn.functional.cosine_similarity(h[k].float(),h[k-10].float(),dim=0,eps=1e-8)
   self.assertAlmostEqual(float(live[-2]),float(expected),places=6)
   other=h.clone();other[k+1:]+=20
   torch.testing.assert_close(live,s.raw_observation(other,k),rtol=0,atol=0)
  broken=full.clone();broken[1,5120:7680]+=1
  with self.assertRaises(ValueError):s.select_profile_features(broken[None])
 def test_cosine_uses_ten_steps_not_previous_token_or_norm(self):
  h=torch.ones(81,2560);h[0,::2]=-1;h[10]=h[0]*3;h[9]=-h[0]
  self.assertAlmostEqual(float(s.raw_observation(h,10)[-2]),0.,places=6)
  self.assertAlmostEqual(float(s.raw_observation(h,20)[-2]),1.,places=6)
 def test_normalization_is_train_only_and_freezes_statistics(self):
  x=torch.randn(32,7,2562);x[...,-2]=torch.linspace(0,2,224).reshape(32,7)
  norm=s.CosineObservationNormalizer().fit(x,source_split="train");got=norm(x)
  torch.testing.assert_close(got[...,:2560],torch.nn.functional.layer_norm(x[...,:2560],(2560,),eps=1e-5))
  self.assertAlmostEqual(float(got[...,-2].mean()),0.,places=6)
  self.assertAlmostEqual(float(got[...,-2].std(unbiased=False)),1.,places=6)
  torch.testing.assert_close(got[...,-1],x[...,-1])
  before={k:v.clone() for k,v in norm.state_dict().items()};norm(x+5)
  self.assertTrue(all(torch.equal(v,norm.state_dict()[k]) for k,v in before.items()))
  with self.assertRaises(ValueError):norm.fit(x,source_split="train")
  with self.assertRaises(ValueError):s.CosineObservationNormalizer().fit(x,source_split="validation")
 def test_identical_network_initial_weights_and_reject_L2_checkpoint(self):
  from controller.simple_state import SimplePPOController
  torch.manual_seed(42);old=SimplePPOController();old.normalizer.fit(torch.randn(2,7,2562),source_split="train")
  torch.manual_seed(42);new=s.CosinePPOController()
  for name,v in old.named_parameters():torch.testing.assert_close(v,dict(new.named_parameters())[name],rtol=0,atol=0)
  with self.assertRaises(ValueError):new.load_state_dict(old.state_dict())
  new.normalizer.fit(torch.randn(2,7,2562),source_split="train")
  with self.assertRaises(ValueError):old.load_state_dict(new.state_dict())
 def test_zero_hidden_and_invalid_inputs(self):
  got=s.raw_observation(torch.zeros(11,2560),10);self.assertTrue(torch.isfinite(got).all());self.assertEqual(float(got[-2]),1)
  with self.assertRaises(ValueError):s.raw_observation(torch.zeros(11,2560),20)
  with self.assertRaises(ValueError):s.CosinePPOController(7685)
 def test_random_init_no_labels_and_fresh_value_critic(self):
  model=make_model();prototype=s.CosineWarmStartController();prototype.normalizer.load_state_dict(model.normalizer.state_dict())
  model.load_warmstart(prototype)
  self.assertTrue((model.critic.weight==0).all());self.assertTrue((model.critic.bias==.5).all())
 def test_old_and_new_checkpoints_not_interchangeable(self):
  model=make_model();old=NormalizedPPOController(7685);old.normalizer.fit(torch.randn(2,7,7685),source_split="train")
  with self.assertRaises(ValueError):model.load_state_dict(old.state_dict())
  with self.assertRaises(ValueError):old.load_state_dict(model.state_dict())
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/"state.pt";torch.save(dict(model=model.state_dict()),path);loaded=s.load_cosine_checkpoint(path)
   x=torch.randn(3,7,2562)
   for a,b in zip(model(x),loaded(x)):torch.testing.assert_close(a,b,rtol=0,atol=0)
 def test_adapters_restore_on_failure(self):
  before=pooled.CohortDecisions,val.CohortDecisions
  with self.assertRaises(RuntimeError):
   with s.live_adapters():
    self.assertIs(pooled.CohortDecisions,s.CosineCohortDecisions);self.assertIs(val.CohortDecisions,s.CosineCohortDecisions)
    raise RuntimeError("deliberate")
  self.assertEqual((pooled.CohortDecisions,val.CohortDecisions),before)
 def test_forced_cap_has_no_policy_forward(self):
  model=make_model();d=s.CosineCohortDecisions(model,2,"cpu",False);d.begin(torch.zeros(2,2560))
  active=torch.ones(2,dtype=torch.bool)
  with patch.object(model,"forward",side_effect=AssertionError("cap forward forbidden")):
   d.step(80,torch.zeros(2,2560),active)
  self.assertEqual(d.x.shape,(2,7,2562));self.assertFalse(d.visited.any());self.assertTrue((d.budgets==80).all())
 def test_protocol_64_fresh_iterations_and_original_ppo_math(self):
  parent=dict(config=dict(warmstart_epochs=0,tie_weight=0,optimization_epochs=4,minibatch_questions=8,learning_rate=3e-4,time_penalty=0,entropy_weight=.01,iterations=32,controller_normalization="old"))
  cfg,checkpoints=protocol(parent,dict(iterations=64,validation_every=16,comparison_iteration=32))
  self.assertEqual(cfg["iterations"],64);self.assertEqual(cfg["optimization_epochs"],4)
  self.assertEqual(checkpoints,[0,16,32,40,48,64])
  self.assertEqual({k for k in cfg if cfg[k]!=parent["config"][k]},{"iterations","controller_normalization"})
 def test_mock_live_A_B_then_16_updates_without_profile_reward(self):
  torch.manual_seed(42);model=make_model();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0)
  ids=[f"fixture:train:{i}" for i in range(32)];sampler=ShuffledQuestionSampler(ids);plan=sampler.plan(1)
  items=[dict(source_id=i,source_split="train") for i in plan["source_ids"]]
  before={k:v.clone() for k,v in model.state_dict().items()};decoded=[]
  def render(wrapper,args,cohort):
   return {},{"A":dict(input_ids=torch.ones(16,1),attention_mask=torch.ones(16,1))}
  def rollout(wrapper,ids,mask,decisions,**kw):
   self.assertIsInstance(decisions,s.CosineCohortDecisions);self.assertTrue(kw["preserve_arithmetic"])
   h=torch.randn(16,2560);decisions.begin(h);active=torch.ones(16,dtype=torch.bool)
   for k in range(1,81):
    h=h+.1;stopped=decisions.step(k,h,active);active=active&~stopped
    if not active.any():break
   return None,[],k
  def decode(wrapper,args,cohort,payloads,budgets,root,path,purpose):
   self.assertEqual(purpose,"live_train_reward")
   self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
   rows=[dict(source_id=v["source_id"],strict_correct=i%2==0) for i,v in enumerate(cohort)]
   decoded.extend(v["source_id"] for v in cohort);write_jsonl(path/"outcomes.jsonl",rows);return rows,0.
  with tempfile.TemporaryDirectory() as directory,s.live_adapters(),patch.object(pooled.torch.cuda,"synchronize"),patch.object(pooled.torch.cuda,"get_rng_state_all",return_value=[]),patch.object(pooled,"ppo_update",side_effect=logged_ppo_update),patch.object(pooled,"reset_cache"),patch.object(pooled,"render_batch",side_effect=render),patch.object(pooled,"online_rollout",side_effect=rollout),patch.object(pooled,"prefixes_for_b",return_value=[]),patch.object(pooled,"decode_cohort",side_effect=decode):
   path=pooled.run_iteration(None,SimpleNamespace(device="cpu",device2="cpu"),Path(directory),items,model,opt,
    torch.full((32,7,2562),float("nan")),torch.zeros(32,7,dtype=torch.bool),CFG,plan,0,auxiliary_pool_size=1024)
   result=json.loads((path/"result.json").read_text())
   self.assertEqual(result["actual_B_completions"],32);self.assertEqual(result["correct"],16)
   self.assertEqual(result["optimizer_updates"],16);self.assertEqual(decoded,plan["source_ids"])
   self.assertTrue(all(float(v["step"])==16 for v in opt.state.values()))
   self.assertTrue(any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
if __name__=="__main__":unittest.main(verbosity=2)
