#!/usr/bin/env python3
"""Isolated B-only greedy diagnostic on archived train32/validation32 prefixes.

512 primary completions plus96 repeat completions (first16/split, K10/40/80).
Never edits frozen profiling/training code; never loads test or trains a model.
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
from controller.profile import (GRID, file_hash, tensor_hash, write_json, write_jsonl,
                                atomic_exclusive_bytes, load_official_medqa)
from scripts.run_matched_controller_ppo import completed, new_attempt, commit, reset_cache, event
from scripts.audit_saved_loop_budgets import repetition
from utils import extract_gsm8k_answer, normalize_answer, medqa_scoring_metadata

REPEAT_GRID = (10, 40, 80)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def score_text(text, gold):
    pred = normalize_answer(extract_gsm8k_answer(text))
    return medqa_scoring_metadata(text, gold, pred, pred == gold)


def reconstruct_payloads(t, reconstruction, k):
    if k not in GRID:
        raise ValueError('Unsupported K')
    payloads = []
    for i in range(16):
        prompt = reconstruction[f'B_prompt_{i}']
        insertion = int(reconstruction[f'insertion_{i}'])
        length = int(t['A_mask'][i].sum())
        prefix = t[f'prefix_{i}'][:length + k]
        if len(prefix) != length + k or not 0 <= insertion <= len(prompt):
            raise ValueError('Invalid archived prefix/insertion')
        payload = torch.cat([prompt[:insertion], prefix, prompt[insertion:]])
        if payload.dtype != torch.bfloat16 or not torch.isfinite(payload).all():
            raise ValueError('Invalid payload precision/finiteness')
        payloads.append({'prompt_embeds': payload})
    return payloads


def check_pairing(payloads, rows, source_ids, split, k):
    if len(payloads) != 16 or len(rows) != 16 or len(source_ids) != 16:
        raise ValueError('Need complete batch16')
    for payload, row, source_id in zip(payloads, rows, source_ids):
        if (row['source_id'] != source_id or row['source_split'] != split
                or row['k'] != k or row['temperature'] != .6 or row['batch'] != 16
                or row['decoder_seed'] != 42 or row['prompt_sha256'] != tensor_hash(payload['prompt_embeds'])):
            raise ValueError('Historical identity/protocol/prompt mismatch')
        if score_text(row['raw_prediction'], row['gold'])['medqa_boxed_v1']['correct'] != row['strict_correct']:
            raise ValueError('Historical strict score mismatch')


def prepare(profile):
    manifest = json.loads((profile / 'manifest.json').read_text())
    finish = json.loads((profile / 'finished.json').read_text())
    verification = json.loads((profile / 'verification.json').read_text())
    if (finish.get('status') != 'complete' or not finish.get('verified')
            or not verification.get('passed') or verification.get('missing')
            or manifest['train_samples'] != 256 or manifest['validation_samples'] != 128):
        raise ValueError('Require verified train256/validation128 profile')
    cfg = manifest['config']
    expected = dict(model_name='Qwen/Qwen3-4B', temperature=.6, generate_bs=16,
                    max_new_tokens=4096, seed=42, num_agents=2, think=False,
                    latent_space_realign=True, vllm_dtype='bfloat16')
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError('Unexpected frozen reference protocol')
    sources = dict(manifest['source_sha256'])
    sources['scripts/pilot_greedy_b.py'] = file_hash(Path(__file__))
    sources['tests/test_greedy_b_pilot.py'] = file_hash(ROOT / 'tests/test_greedy_b_pilot.py')
    for name, digest in sources.items():
        if file_hash(ROOT / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    inputs = {name: file_hash(profile / name) for name in
              ('manifest.json', 'finished.json', 'verification.json', 'shared/realignment.safetensors')}
    selected = {}
    for split in ('train', 'validation'):
        ids = manifest['selections'][split]['source_ids'][:32]
        if len(set(ids)) != 32 or any(f':{split}:' not in i for i in ids):
            raise ValueError('Bad selected cohort')
        selected[split] = ids
        official, metadata = load_official_medqa(cfg['data_dir'], split)
        if metadata['sha256'] != manifest['selections'][split]['dataset_sha256']:
            raise ValueError('Official dataset changed')
        lookup = {i['source_id']: i for i in official}
        for g in range(2):
            teacher = completed(profile / split / 'teachers' / f'cohort{g:03d}')
            if teacher is None:
                raise ValueError('Missing archived teacher')
            meta = json.loads((teacher / 'cohort.json').read_text())
            cohort_ids = ids[g*16:(g+1)*16]
            if meta['source_ids'] != cohort_ids or meta['split'] != split:
                raise ValueError('Teacher cohort changed')
            t = load_file(str(teacher / 'teacher.safetensors'))
            reconstruction = load_file(str(teacher / 'reconstruction.safetensors'))
            for k in GRID:
                historical = completed(profile / split / 'profile' / f'K{k}' / f'cohort{g:03d}')
                if historical is None:
                    raise ValueError('Missing historical B outcomes')
                rows = read_rows(historical / 'outcomes.jsonl')
                check_pairing(reconstruct_payloads(t, reconstruction, k), rows, cohort_ids, split, k)
                if any(row['gold'] != lookup[row['source_id']]['gold'] for row in rows):
                    raise ValueError('Official gold mismatch')
                for p in historical.iterdir():
                    if p.is_file():
                        inputs[str(p.relative_to(profile))] = file_hash(p)
                marker = historical.parent / 'complete.json'
                inputs[str(marker.relative_to(profile))] = file_hash(marker)
            for p in teacher.iterdir():
                if p.is_file():
                    inputs[str(p.relative_to(profile))] = file_hash(p)
            marker = teacher.parent / 'complete.json'
            inputs[str(marker.relative_to(profile))] = file_hash(marker)
            del t, reconstruction
    if set(selected['train']) & set(selected['validation']):
        raise ValueError('Train/validation identity overlap')
    return dict(schema='greedy_B_archived_prefix_pilot_v1', profile=str(profile), config=cfg,
        source_sha256=sources, reference_files_sha256=inputs, selected_source_ids=selected,
        selection='First two frozen cohorts per split; no outcome-based selection',
        grid=list(GRID), greedy=dict(temperature=0., top_p=1., top_k=0, seed=42, max_tokens=4096),
        batch=16, expected_primary_B_completions=512, expected_repeat_B_completions=96,
        repeat=dict(grid=list(REPEAT_GRID), first_cohort_only=True, splits=['train', 'validation']),
        cache='Cold each split/K; warm cohort0 -> cohort1, preserving reference initial history',
        A_forward_calls=0, A_prefix_reused=True, test_loaded=False, optimizer_created=False,
        no_wandb_writes=True, full_repeat_profiling_started=False,
        caveats=['Not a new A trajectory/full-pipeline repeat', 'Historical sampling comparator, not fresh simultaneous sampling',
                 'Only32 questions per split; descriptive pilot, not protocol selection on test',
                 'Greedy repeat covers first16/split at K10/40/80 only'])


@torch.no_grad()
def decode_greedy(engine, payloads, reference_rows, k, params):
    if len(payloads) != 16 or len(reference_rows) != 16 or params.temperature != 0:
        raise ValueError('Require greedy complete batch16')
    started = time.perf_counter()
    outputs = engine.generate(payloads, params, use_tqdm=False)
    seconds = time.perf_counter() - started
    if len(outputs) != 16:
        raise ValueError('Incomplete B outputs')
    rows = []
    for payload, ref, output in zip(payloads, reference_rows, outputs):
        generated = output.outputs[0]
        ids = list(generated.token_ids)
        if not 0 <= len(ids) <= 4096:
            raise ValueError('Invalid B token count')
        metric = score_text(generated.text, ref['gold'])
        strict = metric['medqa_boxed_v1']
        rows.append(dict(source_id=ref['source_id'], source_split=ref['source_split'], k=k,
            gold=ref['gold'], raw_prediction=generated.text, generated_token_ids=ids, output_tokens=len(ids),
            finish_reason=generated.finish_reason, stop_reason=generated.stop_reason,
            strict_prediction=strict['prediction'], strict_correct=strict['correct'], strict_parse_failed=strict['parse_failed'],
            legacy_prediction=metric['legacy_prediction'], legacy_correct=metric['legacy_correct'],
            repetition=repetition(ids), temperature=0., top_p=1., top_k=0, decoder_seed=42, batch=16,
            prompt_sha256=tensor_hash(payload['prompt_embeds']),
            cached_tokens=getattr(output, 'num_cached_tokens', None)))
    return rows, seconds


def tally(rows):
    n = len(rows)
    return dict(samples=n, correct=sum(r['strict_correct'] for r in rows),
        accuracy=sum(r['strict_correct'] for r in rows)/n,
        tokens=sum(r['output_tokens'] for r in rows),
        mean_B_tokens=sum(r['output_tokens'] for r in rows)/n,
        caps=sum(r['finish_reason'] == 'length' for r in rows),
        loops=sum(r['repetition']['loop_flag'] for r in rows),
        parse_failures=sum(r['strict_parse_failed'] for r in rows))


def run_sequence(wrapper, root, frozen, split, k, repeat=False):
    profile = Path(frozen['profile'])
    groups = 1 if repeat else 2
    bases = [root / ('repeat' if repeat else 'primary') / split / f'K{k}' / f'cohort{g:03d}' for g in range(groups)]
    existing = [completed(base) for base in bases]
    if all(p is not None for p in existing):
        return
    missing = next(i for i, p in enumerate(existing) if p is None)
    if any(p is not None for p in existing[missing:]):
        raise ValueError('Nonconsecutive commits')
    reset_cache(wrapper)
    from vllm import SamplingParams
    params = SamplingParams(**frozen['greedy'])
    for g, base in enumerate(bases):
        teacher = completed(profile / split / 'teachers' / f'cohort{g:03d}')
        t = load_file(str(teacher / 'teacher.safetensors'))
        reconstruction = load_file(str(teacher / 'reconstruction.safetensors'))
        payloads = reconstruct_payloads(t, reconstruction, k)
        historical = completed(profile / split / 'profile' / f'K{k}' / f'cohort{g:03d}')
        reference_rows = read_rows(historical / 'outcomes.jsonl')
        check_pairing(payloads, reference_rows,
            frozen['selected_source_ids'][split][g*16:(g+1)*16], split, k)
        replay = g < missing
        if replay:
            base = root / 'cache_replay' / ('repeat' if repeat else 'primary') / split / f'K{k}' / f'cohort{g:03d}'
        path = new_attempt(base)
        for payload in payloads:
            payload['prompt_embeds'] = payload['prompt_embeds'].to(wrapper.HF_device)
        event(root, stage='B_start', split=split, k=k, cohort=g, repeat=repeat, cache_replay=replay)
        rows, seconds = decode_greedy(wrapper.vllm_engine, payloads, reference_rows, k, params)
        if any(a['prompt_sha256'] != b['prompt_sha256'] for a,b in zip(rows, reference_rows)):
            raise ValueError('Payload changed on GPU transfer')
        write_jsonl(path / 'outcomes.jsonl', rows)
        write_json(path / 'summary.json', dict(**tally(rows), seconds=seconds, repeat=repeat, cache_replay=replay))
        if replay:
            write_json(path / 'replay_complete.json', dict(main_completions=0, replay_completions=16))
        else:
            commit(base, path)
        event(root, stage='B_complete', split=split, k=k, cohort=g, repeat=repeat,
            cache_replay=replay, seconds=seconds, **tally(rows))
        del t, reconstruction, payloads


def verify_and_summarize(root, frozen):
    profile = Path(frozen['profile'])
    tables, stability, question_patterns = {}, {}, {}
    for split in ('train', 'validation'):
        tables[split] = {}
        qs = {sid: {'historical': [], 'greedy': []} for sid in frozen['selected_source_ids'][split]}
        for k in GRID:
            old, new = [], []
            for g in range(2):
                ref = completed(profile / split / 'profile' / f'K{k}' / f'cohort{g:03d}')
                current = completed(root / 'primary' / split / f'K{k}' / f'cohort{g:03d}')
                if current is None:
                    raise ValueError('Missing greedy primary cohort')
                olds, news = read_rows(ref / 'outcomes.jsonl'), read_rows(current / 'outcomes.jsonl')
                expected_ids = frozen['selected_source_ids'][split][g*16:(g+1)*16]
                if [r['source_id'] for r in news] != expected_ids:
                    raise ValueError('Greedy identity mismatch')
                for a, b in zip(olds, news):
                    if (a['source_id'] != b['source_id'] or a['prompt_sha256'] != b['prompt_sha256']
                            or b['gold'] != a['gold'] or b['temperature'] != 0 or b['batch'] != 16 or b['k'] != k
                            or b['strict_correct'] != score_text(b['raw_prediction'], b['gold'])['medqa_boxed_v1']['correct']
                            or b['output_tokens'] != len(b['generated_token_ids'])
                            or b['repetition'] != repetition(b['generated_token_ids'])):
                        raise ValueError('Greedy independent verification failed')
                    qs[b['source_id']]['historical'].append(int(a['strict_correct']))
                    qs[b['source_id']]['greedy'].append(int(b['strict_correct']))
                old += olds
                new += news
            tables[split][str(k)] = dict(historical=tally(old), greedy=tally(new),
                correct_to_wrong=[a['source_id'] for a,b in zip(old,new) if a['strict_correct'] and not b['strict_correct']],
                wrong_to_correct=[a['source_id'] for a,b in zip(old,new) if not a['strict_correct'] and b['strict_correct']])
        question_patterns[split] = {}
        for mode in ('historical','greedy'):
            vectors = [v[mode] for v in qs.values()]
            question_patterns[split][mode] = dict(
                all_correct=sum(all(v) for v in vectors), all_wrong=sum(not any(v) for v in vectors),
                mixed=sum(any(v) and not all(v) for v in vectors),
                K10_wrong_later_correct=sum(not v[0] and any(v[1:]) for v in vectors),
                K10_correct_later_wrong=sum(v[0] and not all(v[1:]) for v in vectors),
                per_question={sid: dict(correctness=state[mode]) for sid,state in qs.items()})
        stability[split] = {}
        for k in REPEAT_GRID:
            first = completed(root / 'primary' / split / f'K{k}' / 'cohort000')
            second = completed(root / 'repeat' / split / f'K{k}' / 'cohort000')
            if second is None:
                raise ValueError('Missing greedy repeat')
            a, b = read_rows(first / 'outcomes.jsonl'), read_rows(second / 'outcomes.jsonl')
            if len(b) != 16:
                raise ValueError('Incomplete repeat')
            for x,y in zip(a,b):
                if (x['source_id'] != y['source_id'] or x['prompt_sha256'] != y['prompt_sha256']
                        or y['temperature'] != 0 or y['k'] != k or y['gold'] != x['gold']
                        or y['strict_correct'] != score_text(y['raw_prediction'], y['gold'])['medqa_boxed_v1']['correct']
                        or y['output_tokens'] != len(y['generated_token_ids'])
                        or y['repetition'] != repetition(y['generated_token_ids'])):
                    raise ValueError('Repeat pairing/score mismatch')
            stability[split][str(k)] = dict(samples=16,
                exact_tokens=sum(x['generated_token_ids']==y['generated_token_ids'] for x,y in zip(a,b)),
                same_correctness=sum(x['strict_correct']==y['strict_correct'] for x,y in zip(a,b)),
                changed_correctness_ids=[x['source_id'] for x,y in zip(a,b) if x['strict_correct']!=y['strict_correct']],
                first=tally(a), second=tally(b))
    for name,digest in frozen['reference_files_sha256'].items():
        if file_hash(profile/name) != digest:
            raise ValueError('Reference artifact changed: '+name)
    for name,digest in frozen['source_sha256'].items():
        if file_hash(ROOT/name) != digest:
            raise ValueError('Frozen source changed during job: '+name)
    return dict(status='complete', verified=True, primary_B_completions=512, repeat_B_completions=96,
                total_planned_B_completions=608, tables=tables, stability=stability,
                question_patterns=question_patterns, test_loaded=False, optimizer_updates=0,
                old_artifacts_unchanged=True, scope='B-only paired archived-prefix diagnostic; not controller training')


def markdown(result):
    lines=['# Greedy B pilot: train32 + validation32', '',
           'Cùng archived A prefixes và exact B prompt embeddings; B text, BF16, batch16, cap4096. '
           'Historical sampling0.6 so với greedy0; không phải fresh sampling comparison. '
           'Không dùng test hoặc train controller. Mỗi K có32 câu/split; chỉ descriptive pilot.', '']
    for split, table in result['tables'].items():
        lines += [f'## {split}', '',
            '| K | Sampling đúng /32 | Greedy đúng /32 | Sampling loop/cap | Greedy loop/cap | Mean tokens sampling → greedy | Đúng→sai / Sai→đúng |',
            '|---|---:|---:|---|---|---|---|']
        for k, record in table.items():
            a,b=record['historical'],record['greedy']
            lines.append(f"| {k} | {a['correct']} ({100*a['accuracy']:.2f}%) | {b['correct']} ({100*b['accuracy']:.2f}%) | "
                         f"{a['loops']}/{a['caps']} | {b['loops']}/{b['caps']} | {a['mean_B_tokens']:.1f} → {b['mean_B_tokens']:.1f} | "
                         f"{len(record['correct_to_wrong'])} / {len(record['wrong_to_correct'])} |")
        lines += ['', 'Cross-budget question patterns:', '', '| Decoder | All correct | All wrong | Mixed | K10 sai, K sau đúng | K10 đúng, K sau sai |',
                  '|---|---:|---:|---:|---:|---:|']
        for mode,p in result['question_patterns'][split].items():
            lines.append(f"| {mode} | {p['all_correct']} | {p['all_wrong']} | {p['mixed']} | "
                         f"{p['K10_wrong_later_correct']} | {p['K10_correct_later_wrong']} |")
        lines.append('')
    lines += ['## Greedy repeat stability', '', 'Hai lần trên first16/split, chỉ K10/K40/K80, cold cache trước mỗi sequence. '
              'Giữ nguyên A tensors; không chứng minh full-pipeline hay cross-restart determinism.', '',
              '| Split | K | Exact output token sequences /16 | Same correctness /16 |', '|---|---|---:|---:|']
    for split,table in result['stability'].items():
        for k,r in table.items():
            lines.append(f"| {split} | {k} | {r['exact_tokens']} | {r['same_correctness']} |")
    lines += ['', '## Giới hạn và bước tiếp theo', '',
              '-32 câu/split chưa đủ chốt decoder tốt hơn; không dùng test để chọn protocol.',
              '- Greedy không mặc định bảo đảm hết loop hoặc invariance với batch/cache/numerics.',
              '- Không gộp608 completions thành608 independent questions; chỉ64 câu khác nhau.',
              '- Nếu chuyển protocol chính sang greedy, cần labels greedy và fixed/controller evaluation khớp nhau.',
              '- Hai lượt full profiling sampling0.6 vẫn chưa được khởi chạy bởi pilot này.', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    cli=p.parse_args()
    profile,root=cli.profile.resolve(),cli.out.resolve()
    if ROOT not in root.parents or root==profile or root in profile.parents or profile in root.parents:
        raise ValueError('Use separate output under repo')
    with ExitStack() as stack:
        lock=stack.enter_context((profile/'pipeline.lock').open('r'))
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        frozen=prepare(profile)
        if cli.dry_run:
            print(json.dumps(dict(**frozen,CUDA_initialized=torch.cuda.is_initialized()),indent=2))
            return
        if root.exists():
            if not cli.resume or json.loads((root/'manifest.json').read_text())!=frozen:
                raise ValueError('Need new output or exact --resume')
        else:
            root.mkdir(parents=True)
            write_json(root/'manifest.json',frozen)
            for name in frozen['source_sha256']:
                atomic_exclusive_bytes(root/'source_snapshot'/name,(ROOT/name).read_bytes())
        output_lock=stack.enter_context((root/'pipeline.lock').open('a'))
        fcntl.flock(output_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'finished.json').exists():
            print('Already complete; no GPU initialization')
            return
        os.environ.pop('VLLM_BATCH_INVARIANT',None)
        from models import ModelWrapper
        from utils import set_seed
        set_seed(42)
        args=SimpleNamespace(**dict(frozen['config'],temperature=0.))
        started=time.perf_counter()
        event(root,stage='initializing',planned_B_completions=608,A_forward_calls=0)
        wrapper=ModelWrapper(args.model_name,torch.device(args.device),use_vllm=True,args=args)
        workers=wrapper.vllm_engine.collective_rpc('profile_inspect_weights')
        if any(w['head_dtype']!='torch.bfloat16' or w['embedding_dtype']!='torch.bfloat16'
               or w['quantization'] is not None or w['model_revision']!=args.model_revision for w in workers):
            raise ValueError('B dtype/revision mismatch')
        if wrapper.HF_model.config._commit_hash!=args.model_revision or next(wrapper.HF_model.parameters()).dtype!=torch.bfloat16:
            raise ValueError('HF embedding model mismatch')
        matrix,norm=wrapper._ensure_latent_realign_matrix(wrapper.HF_model,wrapper.HF_device,args)
        archived=load_file(str(profile/'shared/realignment.safetensors'))
        if not torch.equal(matrix.cpu(),archived['matrix']) or not torch.equal(norm.cpu(),archived['target_norm']):
            raise ValueError('Realignment differs')
        write_json(root/f'preflight_{time.time_ns()}.json',dict(workers=workers,realignment_exact=True,
            initialized_seconds=time.perf_counter()-started,torch=torch.__version__,cuda=torch.version.cuda,
            A_forward_calls=0,archived_payloads_verified=512,greedy=frozen['greedy']))
        event(root,stage='engine_ready')
        for split in ('train','validation'):
            for k in GRID:
                run_sequence(wrapper,root,frozen,split,k)
        for split in ('train','validation'):
            for k in REPEAT_GRID:
                run_sequence(wrapper,root,frozen,split,k,repeat=True)
        result=verify_and_summarize(root,frozen)
        result['total_instrumented_seconds']=time.perf_counter()-started
        if not (root/'report.md').exists():
            atomic_exclusive_bytes(root/'report.md',markdown(result).encode())
        if not (root/'comparison.json').exists():
            write_json(root/'comparison.json',result)
        write_json(root/'finished.json',dict(status='complete',verified=True,
            primary_B_completions=512,repeat_B_completions=96,report_sha256=file_hash(root/'report.md'),
            comparison_sha256=file_hash(root/'comparison.json'),total_instrumented_seconds=result['total_instrumented_seconds'],
            test_loaded=False,optimizer_updates=0))
        event(root,stage='pilot_complete',primary_B_completions=512,repeat_B_completions=96,verified=True)


if __name__=='__main__':
    main()
