import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import migrate_b_seed_repeats as migration
from scripts import profile_b_seed_repeats as repeat
from scripts import schedule_b_seed_repeats as scheduler
from scripts import verify_controller_pool as verifier
from scripts.run_matched_controller_ppo import new_attempt, commit
from controller.profile import write_json, write_jsonl


class MigrationTests(unittest.TestCase):
    def test_relocation_keeps_numerical_protocol_and_does_not_mutate_original(self):
        config = dict(data_dir='/old/project/data/medqa', reference='/old/project/output/ref.json',
            worker_extension_cls='controller_profile.ProfileWorkerInspector', model_revision='fixed',
            vllm_dtype='bfloat16', seed=42, temperature=.6, max_new_tokens=4096,
            gpu_memory_utilization=.3, device='cuda:0', device2='cuda:0')
        original = dict(config)
        actual = migration.relocate_config(config, '/new/project', '/new/project/data/medqa', .2, 'cuda:1')
        self.assertEqual(config, original)
        self.assertEqual(actual['reference'], '/new/project/output/ref.json')
        self.assertEqual(actual['worker_extension_cls'], 'controller.profile.ProfileWorkerInspector')
        self.assertEqual(actual['device2'], 'cuda:1')
        for name in ('model_revision', 'vllm_dtype', 'seed', 'temperature', 'max_new_tokens'):
            self.assertEqual(actual[name], config[name])
        with self.assertRaises(ValueError):
            migration.relocate_config(config, '/new/project', '/outside/data')

    def test_historical_sources_checked_in_snapshot_not_moved_worktree(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            snapshot = root/'source_snapshot'
            snapshot.mkdir()
            file = snapshot/'removed_source.py'
            file.write_text('historical')
            expected = {'removed_source.py': hashlib.sha256(file.read_bytes()).hexdigest()}
            migration.verify_hashes(snapshot, expected)
            self.assertFalse((root/'removed_source.py').exists())
            file.write_text('tampered')
            with self.assertRaises(ValueError):
                migration.verify_hashes(snapshot, expected)

    def test_hash_verification_rejects_archive_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                migration.verify_hashes(Path(d), {'../foreign': 'irrelevant'})

    def test_data_root_override_still_enforces_dataset_hash_and_leaves_manifest_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = dict(source_sha256={}, config=dict(data_dir='/old/data'),
                selections={'train': dict(source_ids=[],dataset_sha256='expected')})
            write_json(root/'manifest.json',manifest)
            write_json(root/'preflight_1.json',dict(hf_dtype='torch.bfloat16',realignment_sha256='matrix',workers=[]))
            before=(root/'manifest.json').read_bytes()
            with patch.object(verifier,'file_hash',return_value='matrix'), \
                    patch.object(verifier,'load_official_medqa',return_value=([],dict(sha256='changed'))) as load:
                with self.assertRaisesRegex(ValueError,'Official split checksum changed'):
                    verifier.verify(root,data_root='/new/data')
                load.assert_called_once_with('/new/data','train')
            self.assertEqual((root/'manifest.json').read_bytes(),before)

    def test_preflight_is_bound_to_plan_and_visible_gpus(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            plan=root/'plan.json'
            plan.write_text('{}')
            with self.assertRaisesRegex(ValueError,'explicit destination preflight'):
                migration.require_destination_preflight(plan)
            report=dict(passed=True,plan_sha256=migration.file_hash(plan),grid=[10,20,80],batch=16,
                max_tokens=4096,A_forward_calls=0,runtime={'fixed':'runtime'},cuda_visible_devices='1,4')
            (root/'destination_preflight.json').write_text(json.dumps(report))
            with patch.object(migration,'runtime_fingerprint',return_value={'fixed':'runtime'}), \
                    patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1,4'}):
                migration.require_destination_preflight(plan)
                plan.write_text('{"changed":true}')
                with self.assertRaises(ValueError):
                    migration.require_destination_preflight(plan)

    def test_explicit_gpu_visibility_preserved_and_no_implicit_all_gpu_selection(self):
        migrated=dict(migration={'schema':'migration'},config={'device':'cuda:0','device2':'cuda:1'})
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(scheduler.gpu_visibility('1,4',migrated),'1,4')
            with self.assertRaises(ValueError):
                scheduler.gpu_visibility(None,migrated)
            with self.assertRaises(ValueError):
                scheduler.gpu_visibility('1',migrated)
            with self.assertRaises(ValueError):
                scheduler.gpu_visibility('1,1',migrated)
            self.assertEqual(scheduler.gpu_visibility(None,{}),'0')

    def test_partial_import_validates_batch_and_rejects_holes_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            historical=root/'old'
            profile=root/'reference'
            ids=[f'medqa:train:{i}' for i in range(32)]
            rows=[dict(source_id=sid,source_split='train',gold='a',k=10,prompt_sha256='same',
                decoder_seed=43,engine_seed=42,temperature=.6,top_p=.95,top_k=0,batch=16,
                raw_prediction=r'\boxed{A}',generated_token_ids=[1],output_tokens=1,
                strict_prediction='a',strict_correct=True,strict_parse_failed=False) for sid in ids[:16]]
            oldbase=historical/'seed43/primary/train/K10/cohort000'
            attempt=new_attempt(oldbase)
            write_jsonl(attempt/'outcomes.jsonl',rows)
            commit(oldbase,attempt)
            refbase=profile/'train/profile/K10/cohort000'
            refattempt=new_attempt(refbase)
            write_jsonl(refattempt/'outcomes.jsonl',rows)
            commit(refbase,refattempt)
            marker_before=(oldbase/'complete.json').read_bytes()
            with patch.object(repeat,'COUNTS',dict(train=32)),patch.object(repeat,'GRID',(10,)):
                imported,counts=migration.committed_imports(historical,profile,dict(selected_source_ids={'train':ids}))
                self.assertEqual(counts,{'43':16,'44':0})
                self.assertEqual(len(imported),1)
                self.assertEqual((oldbase/'complete.json').read_bytes(),marker_before)
                # A later seed cannot have commits when seed43 is still incomplete.
                later=historical/'seed44/primary/train/K10/cohort000'
                later_attempt=new_attempt(later)
                write_jsonl(later_attempt/'outcomes.jsonl',rows)
                commit(later,later_attempt)
                with self.assertRaisesRegex(ValueError,'Nonconsecutive historical sequence'):
                    migration.committed_imports(historical,profile,dict(selected_source_ids={'train':ids}))


    def test_renamed_worker_and_separate_process_group_are_tracked_by_identity(self):
        import subprocess
        import sys
        import time
        from scripts.schedule_b_seed_repeats import remember_descendants, live_owned_processes
        child_code = "import setproctitle,time; setproctitle.setproctitle('LatentMASWorkerFixture'); time.sleep(4)"
        parent_code = "import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-B','-c'," + repr(child_code) + "],start_new_session=True); print(child.pid,flush=True); time.sleep(2)"
        parent=subprocess.Popen([sys.executable,'-B','-u','-c',parent_code],cwd=Path(__file__).resolve().parents[1],start_new_session=True,stdout=subprocess.PIPE,text=True)
        worker_pid=int(parent.stdout.readline().strip())
        identities={}
        remember_descendants(parent.pid,identities)
        self.assertIn(worker_pid,identities)
        self.assertIn(worker_pid,[identity['pid'] for identity in live_owned_processes(identities)])
        stale=dict(identities[worker_pid],start_ticks='-1')
        self.assertEqual(live_owned_processes({worker_pid:stale}),[])
        parent.wait(timeout=10)
        self.assertIn(worker_pid,[identity['pid'] for identity in live_owned_processes(identities)])
        deadline=time.monotonic()+8
        while live_owned_processes(identities) and time.monotonic()<deadline:
            time.sleep(.1)
        self.assertEqual(live_owned_processes(identities),[])


    def test_user_parity_waiver_is_explicit_and_bound_without_passing_parity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            authorization = dict(requested_by='user', action='skip_remaining_cross_hardware_parity', request_excerpt='Skip parity and profile now')
            plan = dict(migration=dict(parity_waiver_authorization=authorization,destination_runtime={'fixed':'runtime'}))
            write_json(root/'plan.json',plan)
            report = dict(authorization=authorization, passed=False, parity_skipped=True,
                plan_sha256=migration.file_hash(root/'plan.json'),runtime={'fixed':'runtime'},cuda_visible_devices='1,4')
            write_json(root/'parity_waiver.json',report)
            with patch.object(migration,'runtime_fingerprint',return_value={'fixed':'runtime'}), patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1,4'}):
                actual=migration.require_destination_preflight(root/'plan.json')
                self.assertFalse(actual['passed'])
                self.assertTrue(actual['parity_skipped'])
                self.assertFalse((root/'destination_preflight.json').exists())
                with self.assertRaises(ValueError):
                    scheduler.destination_gate(root,plan,'2,4')
                for field,value in (('passed',True),('plan_sha256','tampered'),('runtime',{'changed':True}),('authorization',{})):
                    (root/'parity_waiver.json').write_text(json.dumps(dict(report,**{field:value})))
                    with self.assertRaises(ValueError):
                        migration.require_destination_preflight(root/'plan.json')
                (root/'parity_waiver.json').write_text(json.dumps(report))
                with patch.object(migration,'runtime_fingerprint',return_value={'changed':True}):
                    with self.assertRaises(ValueError):
                        migration.require_destination_preflight(root/'plan.json')


if __name__=='__main__':
    unittest.main()
