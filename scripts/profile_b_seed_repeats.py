#!/usr/bin/env python3
"""B-only full train256/valid128 repeats, using immutable archived A states.

Only SamplingParams.seed changes (43/44); engine/global seed stays42. No test,
controller training, new A rollout, or changes to the original profile.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file
from controller.profile import (GRID, file_hash, tensor_hash, write_json,
    write_jsonl, atomic_exclusive_bytes, load_official_medqa)
from scripts.run_matched_controller_ppo import completed, new_attempt, commit, reset_cache, event
from scripts.pilot_greedy_b import reconstruct_payloads, check_pairing, score_text, tally, read_rows
from scripts.audit_saved_loop_budgets import repetition

COUNTS = dict(train=256, validation=128)
SEEDS = (43, 44)
NEW_SOURCES = ('scripts/profile_b_seed_repeats.py', 'scripts/schedule_b_seed_repeats.py',
               'scripts/pilot_greedy_b.py', 'tests/test_b_seed_repeats.py')


def sampling_config(seed):
    if seed not in SEEDS:
        raise ValueError('Only approved additional decoder seeds43/44')
    return dict(temperature=.6, top_p=.95, top_k=0, seed=seed, max_tokens=4096)


def check_sources(plan):
    for name, digest in plan['source_sha256'].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    if plan.get('migration'):
        from scripts.migrate_b_seed_repeats import check_execution_sources
        check_execution_sources(plan)


def prepare(profile, out):
    if (ROOT not in out.parents or out == profile or profile in out.parents
            or out in profile.parents):
        raise ValueError('Require dedicated output separate from original profile')
    manifest = json.loads((profile / 'manifest.json').read_text())
    finish = json.loads((profile / 'finished.json').read_text())
    verification = json.loads((profile / 'verification.json').read_text())
    if (finish.get('status') != 'complete' or finish.get('verified') is not True
            or finish.get('main_B_completions') != 3072 or not verification.get('passed')
            or verification.get('missing') or manifest['train_samples'] != 256
            or manifest['validation_samples'] != 128):
        raise ValueError('Require complete verified train256/validation128 reference')
    cfg = manifest['config']
    expected = dict(model_name='Qwen/Qwen3-4B', temperature=.6, top_p=.95,
        generate_bs=16, max_new_tokens=4096, seed=42, num_agents=2, prompt='sequential',
        think=False, latent_space_realign=True, vllm_dtype='bfloat16',
        enable_prefix_caching=True)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('Reference protocol differs')
    sources = dict(manifest['source_sha256'])
    for name in NEW_SOURCES:
        sources[name] = file_hash(ROOT / name)
    check_sources(dict(source_sha256=sources))
    inputs = {name: file_hash(profile / name) for name in
        ('manifest.json', 'finished.json', 'verification.json', 'shared/realignment.safetensors')}
    selections = {}
    checked = 0
    for split, count in COUNTS.items():
        ids = manifest['selections'][split]['source_ids']
        if len(ids) != count or len(set(ids)) != count or any(f':{split}:' not in s for s in ids):
            raise ValueError('Bad cohort identities')
        selections[split] = ids
        official, metadata = load_official_medqa(cfg['data_dir'], split)
        if metadata['sha256'] != manifest['selections'][split]['dataset_sha256']:
            raise ValueError('Dataset changed')
        lookup = {i['source_id']: i for i in official}
        for g in range(count // 16):
            teacher = completed(profile / split / 'teachers' / f'cohort{g:03d}')
            if teacher is None:
                raise ValueError('Missing teacher')
            meta = json.loads((teacher / 'cohort.json').read_text())
            cohort = ids[g*16:(g+1)*16]
            if meta['source_ids'] != cohort or meta['split'] != split:
                raise ValueError('Teacher identity mismatch')
            t = load_file(str(teacher / 'teacher.safetensors'))
            reconstruction = load_file(str(teacher / 'reconstruction.safetensors'))
            for k in GRID:
                historical = completed(profile / split / 'profile' / f'K{k}' / f'cohort{g:03d}')
                if historical is None:
                    raise ValueError('Missing historical outcomes')
                rows = read_rows(historical / 'outcomes.jsonl')
                check_pairing(reconstruct_payloads(t, reconstruction, k), rows, cohort, split, k)
                if any(r['gold'] != lookup[r['source_id']]['gold'] for r in rows):
                    raise ValueError('Gold mismatch')
                checked += len(rows)
                for p in list(historical.iterdir()) + [historical.parent / 'complete.json']:
                    if p.is_file():
                        inputs[str(p.relative_to(profile))] = file_hash(p)
            for p in list(teacher.iterdir()) + [teacher.parent / 'complete.json']:
                if p.is_file():
                    inputs[str(p.relative_to(profile))] = file_hash(p)
            del t, reconstruction
    if set(selections['train']) & set(selections['validation']):
        raise ValueError('Split overlap')
    return dict(schema='B_archived_prefix_full_seed_repeats_v1', profile=str(profile),
        output=str(out), config=cfg, counts=COUNTS, selected_source_ids=selections,
        source_sha256=sources, reference_files_sha256=inputs,
        sampling={str(s): sampling_config(s) for s in SEEDS}, seeds=list(SEEDS),
        engine_seed=42, global_seed=42, grid=list(GRID), batch=16,
        expected_per_seed_completions=3072, expected_new_completions=6144,
        archived_payloads_verified=checked, rest_seconds=3600,
        cache='cold each split/K; warm frozen cohort order; resume replays preceding cohorts',
        A_forward_calls=0, test_loaded=False, optimizer_updates=0, no_wandb_writes=True,
        caveats=['B-only sampling repeats on fixed A; not full-pipeline repeats',
                 'Different seeds do not guarantee independence under every runtime',
                 'Three outcomes per question/K give only a rough success-rate estimate',
                 'No any-of-three correctness reward; no training protocol changed'])


def check_reference(plan):
    check_sources(plan)
    profile = Path(plan['profile'])
    for name, digest in plan['reference_files_sha256'].items():
        if file_hash(profile / name) != digest:
            raise ValueError('Archived artifact changed: ' + name)
    if plan.get('migration'):
        from scripts.migrate_b_seed_repeats import check_migration_lineage
        check_migration_lineage(plan)


@torch.no_grad()
def decode(engine, payloads, refs, seed, params):
    if len(payloads) != 16 or len(refs) != 16:
        raise ValueError('Require batch16')
    expected = sampling_config(seed)
    if any(getattr(params, k) != v for k, v in expected.items()):
        raise ValueError('Decoder protocol mismatch')
    started = time.perf_counter()
    outputs = engine.generate(payloads, params, use_tqdm=False)
    seconds = time.perf_counter() - started
    if len(outputs) != 16:
        raise ValueError('Incomplete B output')
    rows = []
    for payload, ref, output in zip(payloads, refs, outputs):
        generated = output.outputs[0]
        ids = list(generated.token_ids)
        if not 0 <= len(ids) <= 4096:
            raise ValueError('Invalid token count')
        metric = score_text(generated.text, ref['gold'])
        strict = metric['medqa_boxed_v1']
        rows.append(dict(source_id=ref['source_id'], source_split=ref['source_split'],
            k=ref['k'], gold=ref['gold'], raw_prediction=generated.text,
            generated_token_ids=ids, output_tokens=len(ids), finish_reason=generated.finish_reason,
            stop_reason=generated.stop_reason, strict_prediction=strict['prediction'],
            strict_correct=strict['correct'], strict_parse_failed=strict['parse_failed'],
            legacy_prediction=metric['legacy_prediction'], legacy_correct=metric['legacy_correct'],
            repetition=repetition(ids), temperature=.6, top_p=.95, top_k=0,
            decoder_seed=seed, engine_seed=42, batch=16,
            prompt_sha256=tensor_hash(payload['prompt_embeds']),
            cached_tokens=getattr(output, 'num_cached_tokens', None)))
    return rows, seconds


def first_missing(paths):
    existing = [completed(p) is not None for p in paths]
    missing = next((i for i, yes in enumerate(existing) if not yes), len(existing))
    if any(existing[missing:]):
        raise ValueError('Nonconsecutive commits; cannot restore original cache history')
    return missing


def run_sequence(wrapper, root, plan, seed, split, k):
    profile = Path(plan['profile'])
    bases = [root / 'primary' / split / f'K{k}' / f'cohort{g:03d}'
             for g in range(COUNTS[split] // 16)]
    missing = first_missing(bases)
    if missing == len(bases):
        return
    reset_cache(wrapper)
    from vllm import SamplingParams
    params = SamplingParams(**sampling_config(seed))
    for g, base in enumerate(bases):
        teacher = completed(profile / split / 'teachers' / f'cohort{g:03d}')
        t = load_file(str(teacher / 'teacher.safetensors'))
        reconstruction = load_file(str(teacher / 'reconstruction.safetensors'))
        payloads = reconstruct_payloads(t, reconstruction, k)
        historical = completed(profile / split / 'profile' / f'K{k}' / f'cohort{g:03d}')
        refs = read_rows(historical / 'outcomes.jsonl')
        check_pairing(payloads, refs, plan['selected_source_ids'][split][g*16:(g+1)*16], split, k)
        replay = g < missing
        if replay:
            base = root / 'cache_replay' / split / f'K{k}' / f'cohort{g:03d}'
        path = new_attempt(base)
        for payload in payloads:
            target = plan['config']['device'] if plan.get('migration') else wrapper.HF_device
            payload['prompt_embeds'] = payload['prompt_embeds'].to(target)
        event(root, stage='B_start', seed=seed, split=split, k=k, cohort=g, cache_replay=replay)
        rows, seconds = decode(wrapper.vllm_engine, payloads, refs, seed, params)
        if plan.get('migration'):
            for row in rows:
                row['execution_domain'] = 'destination_h100'
                row['historical_plan_sha256'] = plan['migration']['historical_plan_sha256']
        if any(a['prompt_sha256'] != b['prompt_sha256'] for a, b in zip(rows, refs)):
            raise ValueError('GPU transfer changed prompt')
        write_jsonl(path / 'outcomes.jsonl', rows)
        write_json(path / 'summary.json', dict(**tally(rows), seconds=seconds, cache_replay=replay))
        if replay:
            write_json(path / 'replay_complete.json', dict(main_completions=0, replay_completions=16))
        else:
            commit(base, path)
        event(root, stage='B_complete', seed=seed, split=split, k=k, cohort=g,
              cache_replay=replay, seconds=seconds, **tally(rows))
        del t, reconstruction, payloads


def validate_new_rows(rows, refs, seed):
    if len(rows) != 16 or len(refs) != 16:
        raise ValueError('Incomplete saved batch')
    for row, ref in zip(rows, refs):
        for key in ('source_id', 'source_split', 'gold', 'k', 'prompt_sha256'):
            if row[key] != ref[key]:
                raise ValueError('Saved pairing mismatch: ' + key)
        if (row['decoder_seed'] != seed or row['temperature'] != .6 or row['top_p'] != .95
                or row['top_k'] != 0 or row['batch'] != 16 or row['engine_seed'] != 42):
            raise ValueError('Saved decoder mismatch')
        strict = score_text(row['raw_prediction'], row['gold'])['medqa_boxed_v1']
        if (strict['correct'] != row['strict_correct'] or strict['prediction'] != row['strict_prediction']
                or strict['parse_failed'] != row['strict_parse_failed']
                or row['output_tokens'] != len(row['generated_token_ids'])):
            raise ValueError('Saved scorer/token count mismatch')


def verify_seed(directory, plan, seed):
    profile, table = Path(plan['profile']), {}
    total = 0
    for split, count in COUNTS.items():
        table[split] = {}
        for k in GRID:
            rows = []
            for g in range(count // 16):
                path = completed(directory / 'primary' / split / f'K{k}' / f'cohort{g:03d}')
                if path is None:
                    raise ValueError('Missing new cohort')
                ref = completed(profile / split / 'profile' / f'K{k}' / f'cohort{g:03d}')
                batch = read_rows(path / 'outcomes.jsonl')
                validate_new_rows(batch, read_rows(ref / 'outcomes.jsonl'), seed)
                rows.extend(batch)
            if [r['source_id'] for r in rows] != plan['selected_source_ids'][split]:
                raise ValueError('Duplicate/missing cohort question')
            table[split][str(k)] = tally(rows)
            total += len(rows)
    if total != 3072:
        raise ValueError('Wrong completion count')
    check_reference(plan)
    return dict(status='complete', verified=True, decoder_seed=seed, main_B_completions=total,
                tables=table, A_forward_calls=0, test_loaded=False, optimizer_updates=0,
                original_artifacts_unchanged=True)


def run(plan_path, seed, resume=False):
    plan = json.loads(plan_path.read_text())
    root = plan_path.parent / f'seed{seed}'
    if seed not in SEEDS:
        raise ValueError('Invalid seed')
    with ExitStack() as stack:
        parent_lock = stack.enter_context((Path(plan['profile']) / 'pipeline.lock').open('r'))
        fcntl.flock(parent_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if root.exists():
            if not resume:
                raise ValueError('Existing seed output requires --resume')
        else:
            root.mkdir()
            write_json(root / 'manifest.json', dict(plan_sha256=file_hash(plan_path),
                decoder_seed=seed, sampling=sampling_config(seed)))
        if json.loads((root / 'manifest.json').read_text())['plan_sha256'] != file_hash(plan_path):
            raise ValueError('Seed output belongs to a different plan')
        lock = stack.enter_context((root / 'pipeline.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_reference(plan)
        if (root / 'finished.json').exists():
            verify_seed(root, plan, seed)
            print('Already verified complete; no GPU initialization', flush=True)
            return
        if plan.get('migration'):
            from scripts.migrate_b_seed_repeats import require_destination_preflight
            require_destination_preflight(plan_path)
        os.environ.pop('VLLM_BATCH_INVARIANT', None)
        from models import ModelWrapper
        from utils import set_seed
        set_seed(42)
        args = SimpleNamespace(**plan['config'])
        started = time.perf_counter()
        event(root, stage='initializing', seed=seed, planned_B_completions=3072, A_forward_calls=0)
        wrapper = ModelWrapper(args.model_name, torch.device(args.device), use_vllm=True, args=args)
        workers = wrapper.vllm_engine.collective_rpc('profile_inspect_weights')
        if any(w['head_dtype'] != 'torch.bfloat16' or w['embedding_dtype'] != 'torch.bfloat16'
               or w['quantization'] is not None or w['model_revision'] != args.model_revision for w in workers):
            raise ValueError('B precision/revision mismatch')
        if (wrapper.HF_model.config._commit_hash != args.model_revision
                or next(wrapper.HF_model.parameters()).dtype != torch.bfloat16):
            raise ValueError('HF precision/revision mismatch')
        realignment = bind_reference_realignment(wrapper, plan)
        write_json(root / f'preflight_{time.time_ns()}.json', dict(workers=workers,
            **realignment, initialized_seconds=time.perf_counter()-started,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            torch=torch.__version__, cuda=torch.version.cuda, A_forward_calls=0,
            archived_payloads_verified=3072, sampling=sampling_config(seed), engine_seed=42))
        event(root, stage='engine_ready', seed=seed)
        for split in COUNTS:
            for k in GRID:
                run_sequence(wrapper, root, plan, seed, split, k)
        result = verify_seed(root, plan, seed)
        write_json(root / f'verification_{time.time_ns()}.json', result)
        write_json(root / 'finished.json', dict(**result,
            total_instrumented_seconds=time.perf_counter()-started))
        event(root, stage='seed_complete', seed=seed, verified=True, main_B_completions=3072)


def summarize(plan_path):
    plan = json.loads(plan_path.read_text())
    for seed in SEEDS:
        verify_seed(plan_path.parent / f'seed{seed}', plan, seed)
    all_rows, tables = [], {}
    for split, count in COUNTS.items():
        tables[split] = {}
        per_question = {sid: [] for sid in plan['selected_source_ids'][split]}
        for k in GRID:
            batches = {s: [] for s in (42, *SEEDS)}
            for g in range(count // 16):
                for seed in batches:
                    base = (Path(plan['profile']) / split / 'profile' if seed == 42 else
                            plan_path.parent / f'seed{seed}' / 'primary' / split)
                    path = completed(base / f'K{k}' / f'cohort{g:03d}')
                    batches[seed].extend(read_rows(path / 'outcomes.jsonl'))
            buckets = [0]*4
            for i, sid in enumerate(plan['selected_source_ids'][split]):
                correctness = [int(batches[s][i]['strict_correct']) for s in batches]
                successes = sum(correctness)
                buckets[successes] += 1
                record = dict(source_id=sid, source_split=split, k=k, seeds=[42,43,44],
                    correctness=correctness, correct_count=successes, trials=3, success_fraction=successes/3)
                all_rows.append(record)
                per_question[sid].append(successes)
            tables[split][str(k)] = dict(per_seed={str(s): tally(r) for s,r in batches.items()},
                correct_0_1_2_3_of_3=buckets,
                mean_accuracy=sum(i*n for i,n in enumerate(buckets))/(3*count),
                unstable_questions=buckets[1]+buckets[2])
        tables[split]['question_patterns'] = dict(
            never_observed_correct=sum(not any(v) for v in per_question.values()),
            all_K_all_three_correct=sum(all(n==3 for n in v) for v in per_question.values()))
    result = dict(status='complete', verified=True, tables=tables, new_B_completions=6144,
        total_including_historical=9216, test_loaded=False, optimizer_updates=0,
        interpretation='Counts/fractions, not any-of-three correctness rewards or deployment accuracy')
    if plan.get('migration'):
        result.update(migration=plan['migration'],
            historical_imported_B_completions=plan['migration']['imported_B_completions'],
            destination_main_B_completions=plan['expected_destination_completions'],
            seed_only_runtime_controlled_comparison=False)
    write_jsonl(plan_path.parent / 'per_question_K_successes.jsonl', all_rows)
    write_json(plan_path.parent / 'comparison.json', result)
    lines = ['# B seed42/43/44 repeats on fixed archived A', '',
        'Train256 + validation128; temperature0.6, batch16, BF16, cap4096. No test or controller training.',
        'Fractions use three B outcomes, not any-of-three oracle reward. Historical seed42, new43/44.', '']
    if plan.get('migration'):
        lines += ['Mixed GB10/H100 continuation. Imported historical batches are preserved; this is not a seed-only runtime-controlled comparison.', '']
    for split in COUNTS:
        lines += [f'## {split}', '', '| K | Correct42 /43 /44 | Correct0/3 | 1/3 | 2/3 | 3/3 | Mean accuracy |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for k in GRID:
            r = tables[split][str(k)]
            counts = ' / '.join(str(r['per_seed'][str(s)]['correct']) for s in (42,43,44))
            bins = ' | '.join(str(n) for n in r['correct_0_1_2_3_of_3'])
            lines.append(f"| {k} | {counts} | {bins} | {100*r['mean_accuracy']:.2f}% |")
        lines.append('')
    atomic_exclusive_bytes(plan_path.parent / 'report.md', '\n'.join(lines).encode())
    write_json(plan_path.parent / 'finished.json', dict(status='complete', verified=True,
        new_B_completions=6144, comparison_sha256=file_hash(plan_path.parent/'comparison.json'),
        per_question_sha256=file_hash(plan_path.parent/'per_question_K_successes.jsonl'),
        report_sha256=file_hash(plan_path.parent/'report.md')))




def bind_reference_realignment(wrapper, plan):
    matrix, norm = wrapper._ensure_latent_realign_matrix(wrapper.HF_model, wrapper.HF_device,
                                                        SimpleNamespace(**plan['config']))
    archived = load_file(str(Path(plan['profile']) / 'shared/realignment.safetensors'))
    exact = torch.equal(matrix.cpu(), archived['matrix']) and torch.equal(norm.cpu(), archived['target_norm'])
    if not plan.get('migration'):
        if not exact:
            raise ValueError('Realignment mismatch')
    else:
        if (matrix.shape != archived['matrix'].shape or norm.shape != archived['target_norm'].shape
                or not torch.isfinite(archived['matrix']).all() or not torch.isfinite(archived['target_norm']).all()):
            raise ValueError('Invalid archived realignment')
        wrapper._latent_realign_matrices[id(wrapper.HF_model)] = (
            archived['matrix'].to(wrapper.HF_device), archived['target_norm'].to(wrapper.HF_device))
    return dict(realignment_exact=True, archived_realignment_reused=bool(plan.get('migration')),
        destination_recomputed_realignment_exact=exact,
        recomputed_matrix_max_abs_error=float((matrix.cpu()-archived['matrix']).abs().max()),
        recomputed_norm_abs_error=float((norm.cpu()-archived['target_norm']).abs().max()))


@torch.no_grad()
def destination_preflight(plan_path):
    from scripts.migrate_b_seed_repeats import runtime_fingerprint
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    if not plan.get('migration') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Preflight requires a migrated campaign and explicit GPU visibility')
    root = plan_path.parent
    if (root / 'destination_preflight.json').exists():
        raise ValueError('Destination preflight already exists; preserve the prior report')
    with (root / 'destination_preflight.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_reference(plan)
        from models import ModelWrapper
        from utils import set_seed
        from vllm import SamplingParams
        set_seed(42)
        args = SimpleNamespace(**plan['config'])
        wrapper = ModelWrapper(args.model_name, torch.device(args.device), use_vllm=True, args=args)
        workers = wrapper.vllm_engine.collective_rpc('profile_inspect_weights')
        if not workers or any(w['head_dtype'] != 'torch.bfloat16' or w['embedding_dtype'] != 'torch.bfloat16'
                or w['quantization'] is not None or w['model_revision'] != args.model_revision for w in workers):
            raise ValueError('Destination B precision/revision mismatch')
        if (wrapper.HF_model.config._commit_hash != args.model_revision
                or next(wrapper.HF_model.parameters()).dtype != torch.bfloat16):
            raise ValueError('Destination HF precision/revision mismatch')
        realignment = bind_reference_realignment(wrapper, plan)
        params = SamplingParams(temperature=.6, top_p=.95, top_k=0, seed=42, max_tokens=4096)
        teacher = completed(Path(plan['profile']) / 'train/teachers/cohort000')
        tensors = load_file(str(teacher / 'teacher.safetensors'))
        reconstruction = load_file(str(teacher / 'reconstruction.safetensors'))
        results = []
        details = []
        for k in (10, 20, 80):
            reset_cache(wrapper)
            payloads = reconstruct_payloads(tensors, reconstruction, k)
            reference = completed(Path(plan['profile']) / 'train/profile' / f'K{k}/cohort000')
            refs = read_rows(reference / 'outcomes.jsonl')
            check_pairing(payloads, refs, plan['selected_source_ids']['train'][:16], 'train', k)
            for payload in payloads:
                payload['prompt_embeds'] = payload['prompt_embeds'].to(args.device)
            started = time.perf_counter()
            outputs = wrapper.vllm_engine.generate(payloads, params, use_tqdm=False)
            if len(outputs) != 16:
                raise ValueError('Incomplete destination preflight cohort')
            exact = same_score = 0
            for payload, ref, output in zip(payloads, refs, outputs):
                generated = output.outputs[0]
                ids = list(generated.token_ids)
                if tensor_hash(payload['prompt_embeds']) != ref['prompt_sha256'] or not 0 <= len(ids) <= 4096:
                    raise ValueError('Destination prompt/token protocol mismatch')
                strict = score_text(generated.text, ref['gold'])['medqa_boxed_v1']
                exact += ids == ref['generated_token_ids']
                same_score += strict['correct'] == ref['strict_correct']
                details.append(dict(source_id=ref['source_id'], k=k, decoder_seed=42,
                    raw_prediction=generated.text, generated_token_ids=ids,
                    prompt_sha256=ref['prompt_sha256'], strict_correct=strict['correct'],
                    reference_strict_correct=ref['strict_correct'], purpose='destination_subset_parity'))
            row = dict(k=k, samples=16, exact_token_sequences=exact, same_strict_correctness=same_score,
                       seconds=time.perf_counter()-started)
            results.append(row)
            print(json.dumps(dict(stage='destination_subset_parity', **row)), flush=True)
        check_reference(plan)
        report = dict(passed=True, plan_sha256=file_hash(plan_path), grid=[10,20,80], batch=16,
            max_tokens=4096, request_seed=42, engine_seed=42, A_forward_calls=0,
            main_completions=0, diagnostic_completions=48, runtime=runtime_fingerprint(), workers=workers,
            cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            visible_gpus=[dict(logical=i,name=torch.cuda.get_device_name(i),
                total_memory=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())],
            realignment=realignment, comparisons=results,
            protocol_and_payload_checks_passed=True, cross_hardware_equivalence_proven=False,
            interpretation='Structural preflight passed; exact/sample correctness differences are reported, not hidden. Continuation remains mixed-hardware.')
        write_jsonl(root/'destination_preflight_outcomes.jsonl', details)
        write_json(root/'destination_preflight.json', report)
        print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','preflight','run','summarize'])
    p.add_argument('--profile', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--resume', action='store_true')
    cli = p.parse_args()
    root = cli.out.resolve()
    if ROOT not in root.parents:
        raise ValueError('Output must be under repo')
    if cli.action == 'prepare':
        if root.exists() or cli.profile is None:
            raise ValueError('Need fresh output and --profile')
        with (cli.profile.resolve() / 'pipeline.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            plan = prepare(cli.profile.resolve(), root)
        if torch.cuda.is_initialized():
            raise ValueError('Preparation must remain CPU-only')
        root.mkdir(parents=True)
        write_json(root / 'plan.json', plan)
        for name in plan['source_sha256']:
            atomic_exclusive_bytes(root / 'source_snapshot' / name, (ROOT/name).read_bytes())
        print(json.dumps(dict(prepared=True, CUDA_initialized=False,
            archived_payloads_verified=plan['archived_payloads_verified'],
            reference_files=len(plan['reference_files_sha256']), output=str(root))), flush=True)
    elif cli.action == 'preflight':
        destination_preflight(root/'plan.json')
    elif cli.action == 'run':
        if cli.seed is None:
            raise ValueError('--seed required')
        run(root/'plan.json', cli.seed, cli.resume)
    else:
        summarize(root/'plan.json')


if __name__ == '__main__':
    main()
