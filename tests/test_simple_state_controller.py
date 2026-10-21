import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from controller import simple_state as s
from controller.normalized import ObservationNormalizer,NormalizedPPOController
from controller.profile import raw_observation as old_observation,write_jsonl
from controller.sampler import ShuffledQuestionSampler
from controller.training import logged_ppo_update
from scripts import run_pooled_controller_ppo as pooled
from scripts import train_controller_from_profile as val
CFG=dict(optimization_epochs=4,minibatch_questions=8,clip=.2,value_weight=.5,entropy_weight=.01,tie_weight=0,grad_clip=1.)
def make_model():
 x=torch.randn(32,7,2562);model=s.SimplePPOController()
 model.normalizer.fit(x,source_split="train")
 return model
class SimpleStateTests(unittest.TestCase):
 def test_features_match_old_scalar_without_other_blocks_or_future(self):
  h=torch.randn(81,2560).bfloat16()
  for k in [10,20,70]:
   old=old_observation(h,k)
   torch.testing.assert_close(s.raw_observation(h,k),s.select_profile_features(old[None])[0],rtol=0,atol=0)
   other=h.clone();other[k+1:]+=20
   torch.testing.assert_close(s.raw_observation(h,k),s.raw_observation(other,k),rtol=0,atol=0)
  x=torch.randn(2,7,7685);y=x.clone();y[...,2560:7680]+=999;y[...,-5]+=8;y[...,-3:-1]+=9
  torch.testing.assert_close(s.select_profile_features(x),s.select_profile_features(y),rtol=0,atol=0)
 def test_normalization_matches_full_block_and_is_train_only(self):
  x=torch.randn(10,7,7685);small=s.select_profile_features(x)
  full=ObservationNormalizer(7685).fit(x,source_split="train")
  norm=s.SimpleObservationNormalizer().fit(small,source_split="train")
  torch.testing.assert_close(norm(small),s.select_profile_features(full(x)),rtol=1e-5,atol=1e-5)
  with self.assertRaises(ValueError):norm.fit(small,source_split="train")
  with self.assertRaises(ValueError):s.SimpleObservationNormalizer().fit(small,source_split="validation")
 def test_zero_hidden_and_invalid_inputs(self):
  got=s.raw_observation(torch.zeros(11,2560),10);self.assertTrue(torch.isfinite(got).all());self.assertEqual(float(got[-2]),0)
  with self.assertRaises(ValueError):s.raw_observation(torch.zeros(11,2560),20)
  with self.assertRaises(ValueError):s.SimplePPOController(7685)
 def test_random_init_no_labels_and_fresh_value_critic(self):
  model=make_model();prototype=s.SimpleWarmStartController();prototype.normalizer.load_state_dict(model.normalizer.state_dict())
  model.load_warmstart(prototype)
  self.assertTrue((model.critic.weight==0).all());self.assertTrue((model.critic.bias==.5).all())
 def test_old_and_new_checkpoints_not_interchangeable(self):
  model=make_model();old=NormalizedPPOController(7685);old.normalizer.fit(torch.randn(2,7,7685),source_split="train")
  with self.assertRaises(ValueError):model.load_state_dict(old.state_dict())
  with self.assertRaises(ValueError):old.load_state_dict(model.state_dict())
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/"state.pt";torch.save(dict(model=model.state_dict()),path);loaded=s.load_simple_checkpoint(path)
   x=torch.randn(3,7,2562)
   for a,b in zip(model(x),loaded(x)):torch.testing.assert_close(a,b,rtol=0,atol=0)
 def test_adapters_restore_on_failure(self):
  before=pooled.CohortDecisions,val.CohortDecisions
  with self.assertRaises(RuntimeError):
   with s.live_adapters():
    self.assertIs(pooled.CohortDecisions,s.SimpleCohortDecisions);self.assertIs(val.CohortDecisions,s.SimpleCohortDecisions)
    raise RuntimeError("deliberate")
  self.assertEqual((pooled.CohortDecisions,val.CohortDecisions),before)
 def test_forced_cap_has_no_policy_forward(self):
  model=make_model();d=s.SimpleCohortDecisions(model,2,"cpu",False);d.begin(torch.zeros(2,2560))
  active=torch.ones(2,dtype=torch.bool)
  with patch.object(model,"forward",side_effect=AssertionError("cap forward forbidden")):
   d.step(80,torch.zeros(2,2560),active)
  self.assertEqual(d.x.shape,(2,7,2562));self.assertFalse(d.visited.any());self.assertTrue((d.budgets==80).all())
 def test_mock_live_A_B_then_16_updates_without_profile_reward(self):
  torch.manual_seed(42);model=make_model();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0)
  ids=[f"fixture:train:{i}" for i in range(32)];sampler=ShuffledQuestionSampler(ids);plan=sampler.plan(1)
  items=[dict(source_id=i,source_split="train") for i in plan["source_ids"]]
  before={k:v.clone() for k,v in model.state_dict().items()};decoded=[]
  def render(wrapper,args,cohort):
   return {},{"A":dict(input_ids=torch.ones(16,1),attention_mask=torch.ones(16,1))}
  def rollout(wrapper,ids,mask,decisions,**kw):
   self.assertIsInstance(decisions,s.SimpleCohortDecisions);self.assertTrue(kw["preserve_arithmetic"])
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
