#!/usr/bin/env python3
"""Bounded three-seed expansion supervisor; no automatic GPU retries or kills."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from controller.job_gate import atomic_state,sha256,process_identity,identity_alive
from scripts.schedule_b_seed_repeats import remember_descendants,live_owned_processes,rest_remaining


def contained(path):
    p=Path(path).resolve()
    if not p.is_relative_to(ROOT.parent): raise ValueError('Outside permitted workspace')
    return p


def choose_gpu_pair(gpus,utilization,reservation_gib):
    minimum_hf=max(20000,int(reservation_gib*1024)+3072)
    for hf in sorted(gpus,key=lambda x:x['free_mib'],reverse=True):
        for b in sorted(gpus,key=lambda x:x['free_mib'],reverse=True):
            if (hf['index']!=b['index'] and hf['free_mib']>=minimum_hf
                    and b['free_mib']>=max(16000,b['total_mib']*utilization+1200)):
                return [b,hf]
    return None


def gpu_inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free',
                                 '--format=csv,noheader,nounits'],text=True)
    out=[]
    for line in raw.splitlines():
        i,uuid,total,free=[v.strip() for v in line.split(',')]
        out.append(dict(index=int(i),uuid=uuid,total_mib=int(total),free_mib=int(free)))
    return out


def validate_sources(plan):
    for name,h in plan['source_sha256'].items():
        p=contained(ROOT/name)
        if not p.is_relative_to(ROOT) or sha256(p)!=h: raise ValueError('Source changed: '+name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--job-dir',required=True,type=Path)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    root,job=contained(args.out),contained(args.job_dir)
    if not root.is_relative_to(ROOT) or not job.is_relative_to(ROOT.parent/'.remote-workspace/jobs'):
        raise ValueError('Use prepared campaign and task job directory')
    if Path(sys.prefix)!=Path('/data/nvidia/conda_envs/latentmas'):
        raise ValueError('Use selected latentmas environment')
    plan=json.loads((root/'plan.json').read_text())
    plan_hash=sha256(root/'plan.json')
    if (plan['schema']!='medqa_train_pool_expansion_v1' or plan['new_questions']!=768
            or plan['request_seeds']!=[42,43,44] or plan['expected_new_B_completions']!=18432
            or plan['rest_seconds_43_44']!=3600):
        raise ValueError('Unexpected deliberate workload')
    validate_sources(plan)
    workers={};active=None
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=root/'schedule_state.json'
        if state_path.exists():
            if not args.resume: raise ValueError('Existing campaign state requires explicit --resume')
            old=json.loads(state_path.read_text())
            if any(identity_alive(x) for x in old.get('worker_identities',[])):
                raise ValueError('Previous workers remain alive')
            if old['status']=='complete':
                print('Campaign already complete');return
        def emit(status,**values):
            result=dict(status=status,utc=datetime.now(timezone.utc).isoformat(),plan_sha256=plan_hash,
                supervisor_identity=process_identity(os.getpid()),process_group=active,
                worker_identities=list(workers.values()),job=str(job),output=str(root),**values)
            atomic_state(state_path,result);atomic_state(job/'state.json',result)
            print(json.dumps(result),flush=True)
        def check():
            if sha256(root/'plan.json')!=plan_hash: raise ValueError('Prepared plan changed')
            validate_sources(plan)
        def execute(command,phase,env):
            nonlocal active,workers
            check();workers={}
            log=job/f'{phase}_{time.time_ns()}.log'
            with log.open('w') as handle:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                active=child.pid
                identity=process_identity(child.pid)
                metadata=dict(command=command,cwd=str(ROOT),environment=sys.prefix,
                    cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES',''),identity=identity,
                    plan_sha256=plan_hash,log=str(log))
                atomic_state(job/f'{phase}_process.json',metadata)
                last=0.
                while child.poll() is None:
                    remember_descendants(child.pid,workers)
                    if time.monotonic()-last>=15:
                        emit(phase,log=str(log));last=time.monotonic()
                    time.sleep(2)
                rc=child.wait()
            atomic_state(job/f'{phase}_exit.json',dict(exit_code=rc,log=str(log)))
            deadline=time.monotonic()+300
            while live_owned_processes(workers):
                emit('waiting_task_workers_exit',phase=phase,live=live_owned_processes(workers))
                if time.monotonic()>deadline:
                    raise RuntimeError('Recorded workers did not exit; no process killed and no next workload started')
                time.sleep(2)
            active=None
            if rc:
                emit('failed',phase=phase,exit_code=rc,log=str(log))
                raise RuntimeError('Stage failed; no automatic retry')
            check()
        for seed in (42,43,44):
            marker=root/f'seed{seed}/finished.json'
            if marker.exists():
                saved=json.loads(marker.read_text())
                if not saved['verified'] or saved['main_B_completions']!=6144 or saved['plan_sha256']!=plan_hash:
                    raise ValueError('Invalid completed seed')
                continue
            if seed==44:
                start_mono,start_wall=time.monotonic(),time.time()
                while True:
                    remaining=rest_remaining(start_mono,start_wall,time.monotonic(),time.time())
                    emit('resting',next_seed=44,remaining_seconds=remaining,rest_seconds_required=3600)
                    if remaining<=0: break
                    time.sleep(min(30,remaining))
                atomic_state(root/f'rest_release_{time.time_ns()}.json',dict(
                    plan_sha256=plan_hash,elapsed_monotonic=time.monotonic()-start_mono,
                    elapsed_wall=time.time()-start_wall,rest_seconds_required=3600))
            deadline=time.monotonic()+7200
            while True:
                check()
                gpus=gpu_inventory()
                selected=choose_gpu_pair(gpus,plan['config']['gpu_memory_utilization'],
                                          plan['config']['hf_memory_reservation_gib'])
                teacher_count=len(list((root/'new_train/teachers').glob('cohort*/complete.json')))
                disk_required=(plan['new_cohorts']-teacher_count)*plan['teacher_bytes_per_cohort']+plan['disk_reserve_bytes']+1024**3
                free=shutil.disk_usage(ROOT).free
                if selected and free>=disk_required: break
                emit('waiting_resources',seed=seed,disk_free_bytes=free,disk_required_bytes=disk_required,gpus=gpus)
                if time.monotonic()>deadline: raise RuntimeError('Resource gate expired; no GPU run started')
                time.sleep(15)
            mask=','.join(str(g['index']) for g in selected)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=mask,PYTHONDONTWRITEBYTECODE='1',
                     LATENTMAS_PROFILE_EXPANSION_SHA256=plan_hash)
            env.pop('VLLM_BATCH_INVARIANT',None)
            atomic_state(job/f'seed{seed}_resources.json',dict(gpus=selected,cuda_visible_devices=mask,
                disk_free_bytes=free,disk_required_bytes=disk_required,plan_sha256=plan_hash))
            command=[sys.executable,'-B','-u','scripts/profile_controller_expansion.py','run',
                     '--out',str(root),'--seed',str(seed)]
            if (root/f'seed{seed}').exists():
                if not args.resume: raise ValueError('Existing seed data requires deliberate --resume')
                command.append('--resume')
            execute(command,f'seed{seed}',env)
            saved=json.loads(marker.read_text())
            if not saved['verified'] or saved['main_B_completions']!=6144:
                raise ValueError('Seed completion missing/invalid')
        execute([sys.executable,'-B','-u','scripts/profile_controller_expansion.py','finalize',
                 '--out',str(root)],'finalize',dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        finish=json.loads((root/'finished.json').read_text())
        if not finish['verified'] or finish['new_B_completions']!=18432:
            raise ValueError('Invalid campaign completion')
        emit('complete',new_B_completions=18432,total_train=1024,validation=128,
             finished_sha256=sha256(root/'finished.json'))


if __name__=='__main__':
    main()
