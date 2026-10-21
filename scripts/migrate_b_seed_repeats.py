#!/usr/bin/env python3
"""Prepare a versioned cross-server continuation without rewriting old artifacts."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from controller.profile import file_hash, write_json, atomic_exclusive_bytes

EXPECTED_PLAN = 'a89ae78859add66555083b19de03f3cd3a3cd508bd31245c588a8b009d6bd79d'
MIGRATION_SCHEMA = 'B_repeat_cross_server_continuation_v1'


def contained(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if not path.is_relative_to(parent):
        raise ValueError('Path outside archive/workspace: ' + str(path))
    return path


def verify_hashes(root, hashes):
    for name, digest in hashes.items():
        if file_hash(contained(Path(root) / name, root)) != digest:
            raise ValueError('Checksum mismatch: ' + name)


def runtime_fingerprint():
    return dict(architecture=platform.machine(), python=platform.python_version(),
        packages={name: version(name) for name in
                  ('torch', 'transformers', 'vllm', 'safetensors', 'numpy')})


def relocate_config(config, repo, data_root, utilization=None, hf_device=None):
    cfg = dict(config)
    old_repo = Path(config['data_dir']).parent.parent
    for key, value in list(cfg.items()):
        if isinstance(value, str) and Path(value).is_absolute():
            path = Path(value)
            if path.is_relative_to(old_repo):
                cfg[key] = str(contained(Path(repo) / path.relative_to(old_repo), repo))
    cfg['data_dir'] = str(contained(data_root, repo))
    worker = cfg.get('worker_extension_cls')
    if worker not in ('controller_profile.ProfileWorkerInspector', 'controller.profile.ProfileWorkerInspector'):
        raise ValueError('Unknown historical worker import path')
    cfg['worker_extension_cls'] = 'controller.profile.ProfileWorkerInspector'
    if utilization is not None:
        if not 0 < utilization < 1:
            raise ValueError('Invalid GPU utilization')
        cfg['gpu_memory_utilization'] = utilization
    if hf_device is not None:
        if hf_device not in ('cuda:0', 'cuda:1'):
            raise ValueError('Use an explicitly visible local CUDA device')
        cfg['device2'] = hf_device
    return cfg


def historical_inputs(historical, profile, expected_plan=EXPECTED_PLAN):
    historical, profile = Path(historical), Path(profile)
    plan_path = historical / 'plan.json'
    if file_hash(plan_path) != expected_plan:
        raise ValueError('Unexpected historical plan SHA256')
    old = json.loads(plan_path.read_text())
    if old['counts'] != dict(train=256, validation=128) or old['seeds'] != [43, 44]:
        raise ValueError('Unexpected historical scope')
    verify_hashes(historical / 'source_snapshot', old['source_sha256'])
    verify_hashes(profile, old['reference_files_sha256'])
    manifest = json.loads((profile / 'manifest.json').read_text())
    verify_hashes(profile / 'source_snapshot', manifest['source_sha256'])
    return old


def committed_imports(historical, profile, old):
    from scripts.profile_b_seed_repeats import COUNTS, GRID, first_missing, validate_new_rows
    from scripts.run_matched_controller_ppo import completed
    imports, counts = [], {str(seed): 0 for seed in (43, 44)}
    missing_group = False
    for seed in (43, 44):
        for split, count in COUNTS.items():
            for k in GRID:
                paths = [Path(historical) / f'seed{seed}/primary/{split}/K{k}/cohort{g:03d}'
                         for g in range(count // 16)]
                n = first_missing(paths)
                if missing_group and n:
                    raise ValueError('Nonconsecutive historical sequence across split/K/seed')
                if n < len(paths):
                    missing_group = True
                for g, base in enumerate(paths[:n]):
                    attempt = contained(completed(base), historical)
                    ref = completed(Path(profile) / split / 'profile' / f'K{k}/cohort{g:03d}')
                    if ref is None:
                        raise ValueError('Missing reference cohort')
                    rows = [json.loads(s) for s in (attempt / 'outcomes.jsonl').read_text().splitlines()]
                    refs = [json.loads(s) for s in (ref / 'outcomes.jsonl').read_text().splitlines()]
                    validate_new_rows(rows, refs, seed)
                    expected_ids = old['selected_source_ids'][split][g*16:(g+1)*16]
                    if [row['source_id'] for row in rows] != expected_ids:
                        raise ValueError('Imported cohort order changed')
                    marker = base / 'complete.json'
                    imports.append(dict(base=str(base.relative_to(historical)), attempt=attempt.name,
                        marker_sha256=file_hash(marker), checksums=json.loads(marker.read_text())['checksums']))
                    counts[str(seed)] += len(rows)
    return imports, counts


def execution_hashes():
    names = {str(p.relative_to(ROOT)) for directory in ('controller', 'scripts', 'tests', 'methods')
             for p in (ROOT / directory).glob('*.py')}
    names.update(str(p.relative_to(ROOT)) for p in ROOT.glob('*.py'))
    names.update(('requirements-server.txt', 'requirements-tracking.txt'))
    return {name: file_hash(ROOT / name) for name in sorted(names)}


def check_execution_sources(plan):
    migration = plan['migration']
    verify_hashes(Path(plan['output']) / 'source_snapshot', plan['source_sha256'])
    if runtime_fingerprint() != migration['destination_runtime']:
        raise ValueError('Destination runtime changed since migration preparation')


def check_migration_lineage(plan):
    migration = plan['migration']
    if migration['schema'] != MIGRATION_SCHEMA:
        raise ValueError('Unknown migration schema')
    historical = contained(migration['historical_repeat'], ROOT)
    if file_hash(historical / 'plan.json') != migration['historical_plan_sha256']:
        raise ValueError('Historical plan changed')
    old = json.loads((historical / 'plan.json').read_text())
    verify_hashes(historical / 'source_snapshot', old['source_sha256'])
    for batch in migration['imported_commits']:
        for parent in (historical, Path(plan['output'])):
            base = contained(parent / batch['base'], parent)
            if file_hash(base / 'complete.json') != batch['marker_sha256']:
                raise ValueError('Imported commit marker changed')
            verify_hashes(base / batch['attempt'], batch['checksums'])


def require_destination_preflight(plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    if plan.get('migration', {}).get('parity_waiver_authorization') is not None:
        from scripts.schedule_b_seed_repeats import destination_gate
        report = destination_gate(plan_path.parent, plan, os.environ.get('CUDA_VISIBLE_DEVICES'))
        if report.get('runtime') != runtime_fingerprint():
            raise ValueError('Destination runtime changed after user parity waiver')
        return report
    report_path = plan_path.parent / 'destination_preflight.json'
    if not report_path.exists():
        raise ValueError('Run explicit destination preflight before migrated GPU continuation')
    report = json.loads(report_path.read_text())
    if (report.get('passed') is not True or report.get('plan_sha256') != file_hash(plan_path)
            or report.get('grid') != [10, 20, 80] or report.get('batch') != 16
            or report.get('max_tokens') != 4096 or report.get('A_forward_calls') != 0
            or report.get('runtime') != runtime_fingerprint()
            or report.get('cuda_visible_devices') != os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('Destination preflight does not match plan/runtime/GPU visibility')
    return report


def migrate(historical, profile, out, data_root, utilization=None, hf_device=None, parity_waiver_reason=None):
    from scripts.verify_controller_pool import verify
    historical, profile = contained(historical, ROOT), contained(profile, ROOT)
    out, data_root = contained(out, ROOT), contained(data_root, ROOT)
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out) for p in (historical, profile)):
        raise ValueError('Require a fresh output separate from historical archives')
    with ExitStack() as stack:
        for path in (profile / 'pipeline.lock', historical / 'seed43/pipeline.lock'):
            lock = stack.enter_context(path.open('r'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = historical_inputs(historical, profile)
        report = verify(profile, data_root=data_root)
        if not report['passed']:
            raise ValueError('Reference semantic verification failed')
        imports, counts = committed_imports(historical, profile, old)
        cfg = relocate_config(old['config'], ROOT, data_root, utilization, hf_device)
        sources = execution_hashes()
        migration = dict(schema=MIGRATION_SCHEMA, utc=datetime.now(timezone.utc).isoformat(),
            historical_repeat=str(historical), historical_plan_sha256=file_hash(historical/'plan.json'),
            historical_profile_manifest_sha256=file_hash(profile/'manifest.json'),
            historical_source_sha256=old['source_sha256'], destination_runtime=runtime_fingerprint(),
            config_changes={key:dict(old=old['config'].get(key),new=value) for key,value in cfg.items()
                            if value != old['config'].get(key)},
            imported_commits=imports, imported_counts=counts,
            imported_B_completions=sum(counts.values()),
            continuity='Mixed GB10/H100 continuation; not a seed-only runtime-controlled comparison',
            old_process_state_imported=False, historical_artifacts_rewritten=False)
        if parity_waiver_reason is not None:
            if not parity_waiver_reason.strip():
                raise ValueError('Parity waiver requires an explicit user request')
            migration['parity_waiver_authorization'] = dict(requested_by='user',
                action='skip_remaining_cross_hardware_parity', request_excerpt=parity_waiver_reason,
                utc=datetime.now(timezone.utc).isoformat(), parity_passed=False,
                limitation='Cross-hardware numerical equivalence is not established')
        plan = dict(old, schema=MIGRATION_SCHEMA, profile=str(profile), output=str(out), config=cfg,
            source_sha256=sources, migration=migration,
            expected_destination_completions=6144-sum(counts.values()))
        out.mkdir(parents=True)
        write_json(out/'plan.json',plan)
        write_json(out/'migration_verification.json',report)
        for name in sources:
            atomic_exclusive_bytes(out/'source_snapshot'/name,(ROOT/name).read_bytes())
        for seed in (43,44):
            if counts[str(seed)]:
                seed_root=out/f'seed{seed}'
                seed_root.mkdir()
                write_json(seed_root/'manifest.json',dict(plan_sha256=file_hash(out/'plan.json'),
                    decoder_seed=seed,sampling=old['sampling'][str(seed)]))
        for batch in imports:
            source=historical/batch['base']
            destination=out/batch['base']
            destination.mkdir(parents=True)
            shutil.copytree(source/batch['attempt'],destination/batch['attempt'],symlinks=True)
            shutil.copyfile(source/'complete.json',destination/'complete.json')
        check_migration_lineage(plan)
        check_execution_sources(plan)
        print(json.dumps(dict(prepared=True,output=str(out),historical_imported=sum(counts.values()),
            destination_main_remaining=plan['expected_destination_completions'],
            destination_preflight_required=parity_waiver_reason is None,parity_skipped=parity_waiver_reason is not None,CUDA_initialized=False)),flush=True)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical-repeat',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--gpu-memory-utilization',type=float)
    p.add_argument('--hf-device',choices=('cuda:0','cuda:1'))
    p.add_argument('--parity-waiver-reason', help='Exact explicit user request to skip remaining cross-hardware parity; never marks it passed')
    args=p.parse_args()
    migrate(args.historical_repeat,args.profile,args.out,args.data_root,args.gpu_memory_utilization,args.hf_device,args.parity_waiver_reason)


if __name__=='__main__':
    main()
