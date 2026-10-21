#!/usr/bin/env python3
"""CPU-only supervisor: seed43 exits with workers -> rest1h -> seed44.

Does not kill any process, change hardware settings, or automatically retry GPU
failures. On explicit resume after a restart, a fresh full rest hour is required.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from controller.job_gate import atomic_state, sha256, process_identity, identity_alive

REST_SECONDS = 3600


def rest_remaining(start_mono, start_wall, mono, wall):
    return max(0., REST_SECONDS - min(mono-start_mono, wall-start_wall))


def process_group_members(group):
    members = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2]) == group and fields[0] not in ('Z','X'):
                members.append(int(directory.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return members


def verified_finish(root, seed):
    path = root/f'seed{seed}'/'finished.json'
    if not path.exists():
        return False
    finish = json.loads(path.read_text())
    if (finish.get('status') != 'complete' or finish.get('verified') is not True
            or finish.get('decoder_seed') != seed or finish.get('main_B_completions') != 3072):
        raise ValueError('Invalid seed completion marker')
    return True


def validate_plan(root):
    plan = json.loads((root/'plan.json').read_text())
    if (plan['seeds'] != [43,44] or plan['rest_seconds'] != REST_SECONDS
            or plan['expected_new_completions'] != 6144 or plan['counts'] != dict(train=256,validation=128)
            or plan['output'] != str(root) or plan['test_loaded'] or plan['optimizer_updates'] != 0):
        raise ValueError('Only the approved two-seed/rest1h plan is allowed')
    for name, digest in plan['source_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Source changed: '+name)
    return plan


def gpu_visibility(requested, plan):
    value = requested if requested is not None else os.environ.get('CUDA_VISIBLE_DEVICES')
    if value is None:
        if plan.get('migration'):
            raise ValueError('Migrated runs require an explicit GPU visibility list')
        value = '0'
    devices = value.split(',')
    if len(set(devices)) != len(devices) or any(not re.fullmatch(r'(?:[0-9]+|GPU-[0-9a-fA-F-]+)', d) for d in devices):
        raise ValueError('Invalid/duplicate GPU visibility list')
    if plan.get('migration'):
        for key in ('device', 'device2'):
            index = int(plan['config'][key].split(':')[-1])
            if not 0 <= index < len(devices):
                raise ValueError('Runtime device is outside the explicitly visible GPU list')
    return value


def destination_gate(root, plan, visible_devices):
    root = Path(root)
    authorization = plan.get('migration', {}).get('parity_waiver_authorization')
    if authorization is not None:
        report = json.loads((root/'parity_waiver.json').read_text())
        if (authorization.get('requested_by') != 'user'
                or authorization.get('action') != 'skip_remaining_cross_hardware_parity'
                or not authorization.get('request_excerpt')
                or report.get('authorization') != authorization
                or report.get('passed') is not False or report.get('parity_skipped') is not True
                or report.get('plan_sha256') != sha256(root/'plan.json')
                or report.get('runtime') != plan['migration']['destination_runtime']
                or report.get('cuda_visible_devices') != visible_devices):
            raise ValueError('Explicit parity waiver does not match user authorization/plan/runtime/GPU mask')
        return report
    report = json.loads((root/'destination_preflight.json').read_text())
    if report.get('passed') is not True or report.get('plan_sha256') != sha256(root/'plan.json') or report.get('cuda_visible_devices') != visible_devices:
        raise ValueError('Require destination preflight for this plan and GPU mask')
    return report


def campaign_group_members(group, plan_hash):
    marker = ('LATENTMAS_CAMPAIGN_SHA256=' + plan_hash).encode()
    members = []
    for pid in process_group_members(group):
        directory = Path('/proc') / str(pid)
        try:
            if (directory.stat().st_uid == os.getuid()
                    and (directory/'cwd').resolve().is_relative_to(ROOT)
                    and marker in (directory/'environ').read_bytes().split(b'\0')):
                members.append(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return members


def remember_descendants(pid, identities):
    pending = [pid]
    visited = set()
    while pending:
        current_pid = pending.pop()
        if current_pid in visited:
            continue
        visited.add(current_pid)
        identity = process_identity(current_pid)
        if identity is None:
            continue
        directory = Path('/proc') / str(current_pid)
        try:
            if directory.stat().st_uid != os.getuid() or not (directory/'cwd').resolve().is_relative_to(ROOT.parent):
                continue
            identities[current_pid] = identity
            pending.extend(int(value) for value in (directory/'task'/str(current_pid)/'children').read_text().split())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return identities


def live_owned_processes(identities):
    return [identity for identity in identities.values() if identity_alive(identity)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--cuda-visible-devices', help='Physical GPU indices/UUIDs; required for migrated runs')
    cli = p.parse_args()
    root = cli.out.resolve()
    if ROOT not in root.parents:
        raise ValueError('Use prepared dedicated output under repo')
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = validate_plan(root)
        visible_devices = gpu_visibility(cli.cuda_visible_devices, plan)
        plan_hash = sha256(root/'plan.json')
        state_path = root/'state.json'
        if state_path.exists():
            if not cli.resume:
                raise ValueError('Existing schedule requires explicit --resume')
            old = json.loads(state_path.read_text())
            saved_workers = old.get('worker_identities')
            if plan.get('migration') and saved_workers is None:
                raise ValueError('Migrated schedule lacks destination worker identities; inspect before resume')
            if any(identity_alive(identity) for identity in (saved_workers or [])) or (not plan.get('migration') and process_group_members(old.get('process_group', -1))):
                raise ValueError('Old GPU process group still alive; do not duplicate')
            if old.get('status') == 'complete':
                print('Schedule already complete', flush=True)
                return
        active_group = None
        owned_workers = {}
        def emit(status, **fields):
            state = dict(status=status, utc=datetime.now(timezone.utc).isoformat(),
                supervisor_pid=os.getpid(), plan_sha256=plan_hash,
                process_group=active_group, process_group_identity=process_identity(active_group) if active_group else None,
                cuda_visible_devices=visible_devices, worker_identities=list(owned_workers.values()), output=str(root), **fields)
            atomic_state(state_path, state)
            print(json.dumps(state), flush=True)
        env = dict(os.environ)
        env.pop('VLLM_BATCH_INVARIANT', None)
        env.update(PYTHONUNBUFFERED='1', PYTHONHASHSEED='42', OMP_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4', VLLM_NO_USAGE_STATS='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
            TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            CUDA_VISIBLE_DEVICES=visible_devices, LATENTMAS_CAMPAIGN_SHA256=plan_hash)
        if plan.get('migration'):
            destination_gate(root, plan, visible_devices)
        def check():
            if sha256(root/'plan.json') != plan_hash:
                raise ValueError('Plan changed while scheduled')
            validate_plan(root)
            if plan.get('migration'):
                destination_gate(root, plan, visible_devices)
        def command(action, seed=None):
            cmd = [sys.executable,'-u','scripts/profile_b_seed_repeats.py',action,'--out',str(root)]
            if seed is not None:
                cmd += ['--seed',str(seed)]
                if (root/f'seed{seed}').exists():
                    cmd += ['--resume']
            return cmd
        try:
            for seed in (43,44):
                check()
                if not verified_finish(root,seed):
                    log_path = root/f'seed{seed}_launch_{time.time_ns()}.log'
                    with log_path.open('x') as log:
                        child = subprocess.Popen(command('run',seed), cwd=ROOT,env=env,
                            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                            start_new_session=True)
                        active_group = child.pid
                        owned_workers = {}
                        remember_descendants(child.pid, owned_workers)
                        emit('profiling', seed=seed, child_pid=child.pid, log=str(log_path))
                        observed = len(owned_workers)
                        while child.poll() is None:
                            remember_descendants(child.pid, owned_workers)
                            if len(owned_workers) != observed:
                                emit('profiling', seed=seed, child_pid=child.pid, log=str(log_path))
                                observed = len(owned_workers)
                            time.sleep(.5)
                        code = child.returncode
                    deadline = time.monotonic()+180
                    while live_owned_processes(owned_workers):
                        if time.monotonic() >= deadline:
                            raise RuntimeError('GPU workers still alive after child exit; refusing next stage')
                        emit('waiting_for_worker_exit',seed=seed)
                        time.sleep(5)
                    if code != 0 or not verified_finish(root,seed):
                        raise RuntimeError(f'Seed{seed} failed/unfinished exit={code}; no automatic retry')
                if seed == 43:
                    active_group = None
                    owned_workers = {}
                    start_mono, start_wall = time.monotonic(), time.time()
                    while True:
                        check()
                        remaining = rest_remaining(start_mono,start_wall,time.monotonic(),time.time())
                        emit('resting', completed_seed=43, next_seed=44,
                            remaining_rest_seconds=remaining, rest_seconds_required=REST_SECONDS,
                            rest_started_unix=start_wall, not_before_unix=start_wall+REST_SECONDS)
                        if remaining <= 0:
                            break
                        time.sleep(min(30,remaining))
                    release = dict(required_seconds=REST_SECONDS,
                        elapsed_monotonic_seconds=time.monotonic()-start_mono,
                        elapsed_wall_seconds=time.time()-start_wall,
                        seed43_finish_sha256=sha256(root/'seed43'/'finished.json'))
                    atomic_state(root/f'rest_release_{time.time_ns()}.json',release)
            active_group = None
            emit('summarizing')
            with (root/f'summary_{time.time_ns()}.log').open('x') as log:
                subprocess.run(command('summarize'),cwd=ROOT,env=env,check=True,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            if not (root/'finished.json').exists():
                raise RuntimeError('Missing combined verified result')
            emit('complete',new_B_completions=6144,finished_sha256=sha256(root/'finished.json'))
        except BaseException as error:
            emit('failed',error_type=type(error).__name__,error=str(error))
            raise


if __name__ == '__main__':
    main()
