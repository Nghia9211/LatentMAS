from datetime import datetime, timezone
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from controller.job_gate import RestGate, REST_SECONDS, profile_ready, check_release, identity_alive, process_identity
from controller.profile import write_json, file_hash
from controller.ppo import PPOController
from controller.warmstart import WarmStartController
from scripts import train_controller_from_profile as trainer


class GateTests(unittest.TestCase):
    def test_wait_for_verified_completion_AND_process_exit_then_full_three_hours(self):
        gate = RestGate()
        for ready, busy in ((False, True), (True, True), (False, False)):
            self.assertEqual(gate.step(ready, busy, 100., 1000.)[0], 'waiting_for_profile')
        self.assertEqual(gate.step(True, False, 101., 1001.), ('resting', 10800.))
        self.assertEqual(gate.step(True, False, 10900., 11800.)[0], 'resting')
        self.assertEqual(gate.step(True, False, 10901., 11801.), ('released', 0.))

    def test_clock_jumps_or_renewed_profile_activity_cannot_shorten_rest(self):
        gate = RestGate()
        gate.step(True, False, 10., 100.)
        self.assertEqual(gate.step(True, False, 20., 20000.)[0], 'resting')
        self.assertEqual(gate.step(True, False, 11000., 90.)[0], 'resting')
        gate.step(True, True, 12000., 12000.)
        self.assertEqual(gate.step(True, False, 13000., 13000.), ('resting',10800.))

    def test_pid_reuse_and_zombie_do_not_block_rest(self):
        old = dict(pid=123, start_ticks='10', state='R')
        with patch('controller.job_gate.process_identity', return_value=dict(pid=123,start_ticks='11',state='R')):
            self.assertFalse(identity_alive(old))
        with patch('controller.job_gate.process_identity', return_value=dict(pid=123,start_ticks='10',state='Z')):
            self.assertFalse(identity_alive(old))
        with patch('controller.job_gate.process_identity', return_value=old):
            self.assertTrue(identity_alive(old))

    def test_incomplete_or_failed_profile_never_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertIsNone(profile_ready(root))
            write_json(root/'manifest.json',dict(train_samples=256,validation_samples=128))
            write_json(root/'finished.json',dict(status='complete',verified=True,train=256,validation=128,
                main_B_completions=3072,training_started=False))
            write_json(root/'verification.json',dict(passed=False,missing=[]))
            with self.assertRaises(ValueError):
                profile_ready(root)

    def test_exact_job_release_and_three_hour_proof_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            profile=root/'profile';profile.mkdir()
            write_json(profile/'manifest.json',dict(train_samples=256,validation_samples=128))
            write_json(profile/'finished.json',dict(status='complete',verified=True,train=256,validation=128,
                main_B_completions=3072,training_started=False))
            write_json(profile/'verification.json',dict(passed=True,missing=[]))
            plan=root/'plan.json'
            write_json(plan,dict(iterations=80))
            source='scripts/train_controller_from_profile.py'
            release=dict(profile=str(profile),training_output=str(root/'training'),iterations=80,
                rest_seconds_required=10800,elapsed_monotonic_seconds=10800.,elapsed_wall_seconds=10800.,
                not_before_unix=20000.,profile_sha256=profile_ready(profile),profiling_processes=[],
                plan_file=str(plan),plan_sha256=file_hash(plan),
                source_sha256={source:file_hash(trainer.ROOT/source)})
            write_json(root/'release.json',release)
            self.assertEqual(check_release(root/'release.json',profile,root/'training',now=20000.)['iterations'],80)
            for output, now in ((root/'different',20000.),(root/'training',19999.)):
                with self.assertRaises(ValueError):
                    check_release(root/'release.json',profile,output,now=now)
            with patch('controller.job_gate.identity_alive',return_value=True):
                release['profiling_processes']=[{'pid':1,'start_ticks':'x'}]
                # Write a separate immutable release fixture.
                write_json(root/'busy_release.json',release)
                with self.assertRaises(ValueError):
                    check_release(root/'busy_release.json',profile,root/'training',now=20000.)


class TrainingTests(unittest.TestCase):
    def test_validation_is_never_accepted_as_training_tensor_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'fake';path.mkdir()
            write_json(path/'manifest.json',dict(split='validation',source_ids=[],allowed_for_gradient_updates=False))
            with patch.object(trainer,'completed',return_value=path), patch.object(trainer,'load_file') as read:
                with self.assertRaises(ValueError):
                    trainer.training_tensors(root,[])
                read.assert_not_called()

    def test_warmstart_all256_and_new_critic_optimizer_without_old_checkpoint(self):
        torch.manual_seed(42)
        model=PPOController(9,16)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0)
        d=dict(features=torch.randn(256,7,9),actions=torch.zeros(256,7,dtype=torch.long),
               q_targets=torch.ones(256,7,2))
        class Tracker:
            rows=[]
            def log(self,row):
                self.rows.append(row)
        tracker=Tracker()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(trainer,'WarmStartController',side_effect=lambda *a: WarmStartController('raw_mlp',9,16)), \
                patch.object(trainer.torch.cuda,'synchronize'), \
                patch.object(trainer.torch.cuda,'get_rng_state_all',return_value=[]):
            root=Path(directory)
            path=trainer.warmstart(model,optimizer,d,dict(device='cpu',learning_rate=.0003,
                grad_clip=1.,warmstart_epochs=50),root,tracker)
            saved=torch.load(path,weights_only=True)
            self.assertEqual(saved['iteration'],0)
            self.assertFalse(saved['optimizer']['state'])
            self.assertTrue(saved['model']['critic.weight'].eq(0).all())
            self.assertTrue(saved['model']['critic.bias'].eq(.5).all())
            self.assertEqual(len(tracker.rows),50)
            result=json.loads((path.parent/'result.json').read_text())
            self.assertEqual(result['train_questions'],256)
            self.assertFalse(result['old_checkpoint_loaded'])

    def test_validation_generates_only128_and_preserves_weights_and_rng(self):
        torch.manual_seed(42)
        model=torch.nn.Linear(2,2).train()
        before={k:v.clone() for k,v in model.state_dict().items()}
        torch_rng=torch.get_rng_state().clone(); python_rng=random.getstate(); numpy_rng=np.random.get_state()
        items=[dict(source_id=f'x:validation:{i}',source_split='validation') for i in range(128)]
        encoded={'A':dict(input_ids=torch.ones(16,2,dtype=torch.long),attention_mask=torch.ones(16,2,dtype=torch.long))}
        def decode(wrapper,args,cohort,payloads,budgets,root,path,purpose):
            self.assertEqual(purpose,'validation_PPO')
            random.random();np.random.rand();torch.rand(1)
            return [dict(source_id=i['source_id'],source_split='validation',k=10,strict_correct=True,
                output_tokens=10,finish_reason='stop',strict_parse_failed=False,repetition={'loop_flag':False})
                for i in cohort],.1
        decisions=SimpleNamespace(budgets=torch.full((16,),10),x=torch.zeros(16,7,9),visited=torch.ones(16,7,dtype=torch.bool))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(trainer,'render_batch',return_value=({},encoded)), \
                patch.object(trainer,'CohortDecisions',return_value=decisions), \
                patch.object(trainer,'online_rollout',return_value=(None,[],10)), \
                patch.object(trainer,'prefixes_for_b',return_value=[{}]*16), \
                patch.object(trainer,'reset_cache') as resets, \
                patch.object(trainer,'decode_cohort',side_effect=decode) as calls:
            metrics=trainer.validation(None,SimpleNamespace(device='cpu',device2='cpu'),model,items,
                Path(directory),16,{'10':{'accuracy':.75}})
            self.assertEqual(calls.call_count,8)
            self.assertEqual(resets.call_count,1)
            self.assertEqual(metrics['validation/PPO/accuracy'],1.)
            # Re-reading the committed validation must not generate again.
            self.assertEqual(trainer.validation(None,None,model,items,Path(directory),16,{}),metrics)
            self.assertEqual(calls.call_count,8)
        self.assertTrue(model.training)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(torch.equal(torch.get_rng_state(),torch_rng))
        self.assertEqual(random.getstate(),python_rng)
        self.assertTrue(np.array_equal(np.random.get_state()[1],numpy_rng[1]))


if __name__=='__main__':
    unittest.main()
