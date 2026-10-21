#!/usr/bin/env python3
"""Profile official train256/validation128, batch16, temp0.6, K10..80.

No optimizer/controller training or historical test300 evaluation. Immutable
cohort attempts, source/data locks, full all-layer trajectories and independent
CPU verification. The historical matched train32 source remains unchanged.
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
from controller.profile import (GRID, file_hash, write_json, write_tensors,
    atomic_exclusive_bytes, raw_observation, phi4, normalized_question)
from controller.sampler import train_pool
from controller.matched import prefixes_for_b, online_rollout
from controller.warmstart import supervision
from scripts.validate_controller_wandb import validation_items
from scripts.profile_controller import feature_schema
from scripts.run_matched_controller_ppo import (SOURCES as BASE_SOURCES, capture_teacher,
    read_teacher, completed, new_attempt, commit, reset_cache, decode_cohort, event)
from latent_batch import rollout

SOURCES = list(BASE_SOURCES) + ['controller/sampler.py', 'scripts/profile_controller_pool.py',
    'scripts/validate_controller_wandb.py', 'scripts/profile_controller.py',
    'scripts/verify_controller_profile.py', 'scripts/verify_controller_pool.py']


def select_splits(data_dir, required_ids, train_count=256, validation_count=128):
    train, train_meta = train_pool(data_dir, train_count, 42, required_ids)
    validation, validation_meta = validation_items(data_dir, validation_count, 42)
    train_stems = {normalized_question(i['question_stem']) for i in train}
    if train_stems & {normalized_question(i['question_stem']) for i in validation}:
        raise ValueError('Train/validation question overlap')
    return {'train': train, 'validation': validation}, {'train': train_meta, 'validation': validation_meta}


def checked_cohort(items, split):
    if (len(items) != 16 or split not in ('train', 'validation')
            or any(i['source_split'] != split or f':{split}:' not in i['source_id'] for i in items)):
        raise ValueError('Need sixteen ordered examples of the declared split, never test')


@torch.no_grad()
def capture(wrapper, args, items, path, split):
    checked_cohort(items, split)
    t, texts, encoded = capture_teacher(wrapper, args, items, path)
    # Rich tensors keep all layers/steps. These are merely causal derived views.
    h = t['hidden'][:, :, -1]
    x = torch.stack([torch.stack([raw_observation(row, k) for k in GRID]) for row in h])
    phi = torch.stack([torch.stack([phi4(row, k) for k in GRID]) for row in h])
    write_tensors(path / 'feature_grid.safetensors', dict(features=x, phi4=phi))
    ids = encoded['B']['input_ids'].to(wrapper.HF_device)
    embeds = wrapper.embedding_layer(ids)
    reconstruction = {}
    marker = '<|im_start|>user\n'
    for i, text in enumerate(texts['B']):
        at = text.find(marker)
        if at < 0:
            raise ValueError('Missing prefix insertion marker')
        insertion = len(wrapper.tokenizer(text[:at + len(marker)], add_special_tokens=False)['input_ids'])
        reconstruction[f'B_prompt_{i}'] = embeds[i, encoded['B']['attention_mask'][i].to(embeds.device).bool()]
        reconstruction[f'insertion_{i}'] = torch.tensor(insertion, dtype=torch.long)
    write_tensors(path / 'reconstruction.safetensors', reconstruction)
    write_json(path / 'cohort.json', dict(split=split, source_ids=[i['source_id'] for i in items],
        hidden_shape=list(t['hidden'].shape), latent_shape=list(t['latent'].shape),
        checkpoint_grid=list(GRID), metadata=['k/80'], labels_in_features=False))


@torch.no_grad()
def check_parity(wrapper, args, teacher_path, path):
    t, _, encoded = read_teacher(teacher_path)
    ids = encoded['A']['input_ids'].to(args.device2)
    mask = encoded['A']['attention_mask'].to(args.device2)
    rows = []
    # Fresh fixed-K rollouts: detect collector/slicing/position errors BEFORE B.
    for k in GRID:
        past, prefixes = rollout(wrapper, ids, mask, latent_steps=k)
        del past
        for i, actual in enumerate(prefixes):
            expected = t[f'prefix_{i}'][:int(mask[i].sum()) + k]
            exact = torch.equal(actual.cpu(), expected)
            rows.append(dict(row=i, k=k, exact=exact,
                max_abs_diff=float((actual.cpu().float() - expected.float()).abs().max())))
        del prefixes
    class Forced:
        budgets = torch.tensor(list(GRID) * 2, device=args.device2)
        def begin(self, h):
            pass
        def step(self, k, h, active):
            return active & (self.budgets == k)
    past, prefixes, _ = online_rollout(wrapper, ids, mask, Forced(), preserve_arithmetic=True)
    del past
    for i, actual in enumerate(prefixes):
        k = int(Forced.budgets[i])
        expected = t[f'prefix_{i}'][:int(mask[i].sum()) + k]
        rows.append(dict(row=i, k=k, mixed=True, exact=torch.equal(actual.cpu(), expected),
            max_abs_diff=float((actual.cpu().float() - expected.float()).abs().max())))
    report = dict(passed=all(r['exact'] for r in rows), rows=rows,
        fixed_cases=128, mixed_cases=16, B_extra_generations=0)
    write_json(path, report)
    if not report['passed']:
        raise RuntimeError('Prefix parity failed; HOLD before downstream label collection')


def sequence_progress(root, split, k, groups):
    paths = [completed(root / split / 'profile' / f'K{k}' / f'cohort{g:03d}') for g in range(groups)]
    first_missing = next((g for g, p in enumerate(paths) if p is None), groups)
    if any(p is not None for p in paths[first_missing:]):
        raise ValueError('Non-consecutive B sequence commits')
    return paths, first_missing


def profile_split(wrapper, args, items, root, split):
    groups = len(items) // 16
    teachers = []
    for g in range(groups):
        cohort = items[g * 16:(g + 1) * 16]
        checked_cohort(cohort, split)
        base = root / split / 'teachers' / f'cohort{g:03d}'
        path = completed(base)
        if path is None:
            path = new_attempt(base)
            event(root, stage='A_capture_start', split=split, cohort=g, total_cohorts=groups)
            capture(wrapper, args, cohort, path, split)
            commit(base, path)
            event(root, stage='A_capture_complete', split=split, cohort=g)
        if json.loads((path / 'cohort.json').read_text())['source_ids'] != [i['source_id'] for i in cohort]:
            raise ValueError('Saved teacher cohort identity changed')
        teachers.append(path)
    parity = root / split / 'prefix_parity.json'
    if not parity.exists():
        event(root, stage='prefix_parity_start', split=split)
        check_parity(wrapper, args, teachers[0], parity)
    if not json.loads(parity.read_text())['passed']:
        raise RuntimeError('Previous parity failed')
    event(root, stage='prefix_parity_passed', split=split, fixed_cases=128, mixed_cases=16)
    for k in GRID:
        paths, first_missing = sequence_progress(root, split, k, groups)
        if first_missing == groups:
            continue
        # Each full split/K sequence is cold at its start, warm between cohorts.
        # On resume replay ALL previously committed cohorts in order to recover
        # cache history; replay outcomes/calls are separate, not free or relabeled.
        reset_cache(wrapper)
        for g in range(groups):
            t, texts, encoded = read_teacher(teachers[g])
            cohort = items[g * 16:(g + 1) * 16]
            prefixes = [t[f'prefix_{i}'][:int(encoded['A']['attention_mask'][i].sum()) + k].to(args.device2)
                        for i in range(16)]
            payloads = prefixes_for_b(wrapper, texts, encoded, prefixes)
            replay = g < first_missing
            base = (root / split / 'cache_replay' / f'K{k}' / f'cohort{g:03d}' if replay else
                    root / split / 'profile' / f'K{k}' / f'cohort{g:03d}')
            path = new_attempt(base)
            purpose = ('train_warmstart_grid' if split == 'train' else 'validation_full_grid')
            if replay:
                purpose = split + '_profile_cache_replay'
            event(root, stage='profile_cohort_start', split=split, k=k, cohort=g, cache_replay=replay)
            rows, seconds = decode_cohort(wrapper, args, cohort, payloads, [k] * 16, root, path, purpose)
            if ([r['source_id'] for r in rows] != [i['source_id'] for i in cohort]
                    or any(r['source_split'] != split for r in rows)):
                raise ValueError('B outcomes mismatch declared split/cohort')
            write_json(path / 'result.json', dict(split=split, k=k, cohort=g, episodes=16,
                cache_replay=replay, seconds=seconds, correct=sum(r['strict_correct'] for r in rows)))
            if replay:
                # Replay attempts stay immutable but are not main-grid commits.
                write_json(path / 'replay_complete.json', dict(episodes=16, counts_as_main=False))
            else:
                commit(base, path)
            event(root, stage='profile_cohort_complete', split=split, k=k, cohort=g,
                cache_replay=replay, main_B_completions=(0 if replay else 16), seconds=seconds)
            del t, prefixes, payloads


def export_dataset(root, items, split):
    grids, correctness = [], []
    for g in range(len(items) // 16):
        teacher = completed(root / split / 'teachers' / f'cohort{g:03d}')
        grids.append(load_file(str(teacher / 'feature_grid.safetensors'))['features'][:, :7])
        q = []
        for k in GRID:
            path = completed(root / split / 'profile' / f'K{k}' / f'cohort{g:03d}')
            rows = [json.loads(s) for s in (path / 'outcomes.jsonl').read_text().splitlines()]
            cohort = items[g * 16:(g + 1) * 16]
            if ([r['source_id'] for r in rows] != [i['source_id'] for i in cohort]
                    or any(r['source_split'] != split or r['k'] != k for r in rows)):
                raise ValueError('Export split/cohort/grid mismatch')
            q.append(torch.tensor([float(r['strict_correct']) for r in rows]))
        correctness.append(torch.stack(q, dim=1))
    x, q = torch.cat(grids), torch.cat(correctness)
    tensors = dict(features=x, correctness=q)
    if split == 'train':
        actions, targets = zip(*(supervision(row) for row in q))
        tensors.update(actions=torch.stack(actions), q_targets=torch.stack(targets))
    base = root / split / 'dataset'
    existing = completed(base)
    if existing is not None:
        return existing
    path = new_attempt(base)
    write_tensors(path / 'dataset.safetensors', tensors)
    write_json(path / 'manifest.json', dict(split=split, source_ids=[i['source_id'] for i in items],
        questions=len(items), features_shape=list(x.shape), correctness_shape=list(q.shape),
        file_sha256=file_hash(path / 'dataset.safetensors'), allowed_for_gradient_updates=split == 'train',
        labels='Single sampled B outcome per fixed-cohort checkpoint, not expected correctness',
        policy_inputs_exclude_future_gold_B_outputs_timing=True))
    commit(base, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-samples', type=int, default=256)
    parser.add_argument('--validation-samples', type=int, default=128)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args()
    root, parent_path = cli.out.resolve(), cli.parent.resolve()
    if ROOT not in root.parents or parent_path in root.parents or root in parent_path.parents or root == parent_path:
        raise ValueError('Use a dedicated output under the repo, separate from the parent')
    with ExitStack() as stack:
        parent_lock = stack.enter_context((parent_path / 'pipeline.lock').open('r'))
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = json.loads((parent_path / 'manifest.json').read_text())
        for name, digest in parent['source_sha256'].items():
            if file_hash(ROOT / name) != digest:
                raise ValueError('Frozen parent source changed: ' + name)
        cfg = dict(parent['config'], train_samples=cli.train_samples,
            validation_samples=cli.validation_samples, test_samples=0,
            test_cohort='not_used_for_profiling', iterations=0)
        lock = json.loads((parent_path / 'training_lock.json').read_text())
        if lock['arithmetic'] != 'preserve':
            raise ValueError('Require approved preserve arithmetic')
        items, selections = select_splits(cfg['data_dir'], lock['source_ids'],
                                         cli.train_samples, cli.validation_samples)
        schema = feature_schema()
        frozen = dict(config=cfg, selections=selections, feature_schema=schema,
            source_sha256={name: file_hash(ROOT / name) for name in SOURCES},
            parent=str(parent_path), train_samples=cli.train_samples, validation_samples=cli.validation_samples,
            expected_main_B_completions=8 * sum(len(v) for v in items.values()),
            extra_B_parity_generations=0, old_profile_outcomes_reused=False,
            checkpoints=list(GRID), training_started=False, historical_test300_used=False,
            protocol=dict(temperature=.6, top_p=.95, top_k=0, seed=42, B_cap=4096, batch=16,
                A_all_layer_every_step=True, cache='cold_each_split_K_sequence_warm_between_cohorts',
                precision='BF16 A/B/head, unchanged realignment',
                cohort_order='Frozen pool manifest order; not future shuffled PPO cohorts'))
        if root.exists():
            if not cli.resume or json.loads((root / 'manifest.json').read_text()) != frozen:
                raise ValueError('Use a new output or unchanged --resume manifest')
        elif cli.resume:
            raise ValueError('No prepared run to resume')
        if cli.dry_run:
            print(json.dumps(frozen, indent=2))
            return
        if not root.exists():
            root.mkdir(parents=True)
            write_json(root / 'manifest.json', frozen)
            for name in SOURCES:
                atomic_exclusive_bytes(root / 'source_snapshot' / name, (ROOT / name).read_bytes())
        output_lock = stack.enter_context((root / 'pipeline.lock').open('a'))
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'finished.json').exists():
            print('Profiling already complete; no GPU initialization')
            return
        os.environ.pop('VLLM_BATCH_INVARIANT', None)
        from models import ModelWrapper
        from utils import set_seed
        args = SimpleNamespace(**cfg)
        set_seed(42)
        started = time.perf_counter()
        event(root, stage='initializing', main_B_completions=frozen['expected_main_B_completions'])
        wrapper = ModelWrapper(args.model_name, torch.device(args.device), use_vllm=True, args=args)
        workers = wrapper.vllm_engine.collective_rpc('profile_inspect_weights')
        if any(w['head_dtype'] != 'torch.bfloat16' or w['embedding_dtype'] != 'torch.bfloat16'
                or w['quantization'] is not None or w['model_revision'] != args.model_revision for w in workers):
            raise ValueError('B precision/revision mismatch')
        if (wrapper.HF_model.config._commit_hash != args.model_revision
                or next(wrapper.HF_model.parameters()).dtype != torch.bfloat16):
            raise ValueError('A precision/revision mismatch')
        matrix, target = wrapper._ensure_latent_realign_matrix(wrapper.HF_model, wrapper.HF_device, args)
        shared = root / 'shared' / 'realignment.safetensors'
        if not shared.exists():
            write_tensors(shared, dict(matrix=matrix, target_norm=target))
        else:
            saved = load_file(str(shared))
            if not torch.equal(saved['matrix'], matrix.cpu()) or not torch.equal(saved['target_norm'], target.cpu()):
                raise ValueError('Realignment changed on resume')
        write_json(root / f'preflight_{time.time_ns()}.json', dict(workers=workers,
            hf_dtype='torch.bfloat16', realignment_sha256=file_hash(shared),
            gpu=torch.cuda.get_device_name(), initialized_seconds=time.perf_counter() - started,
            torch=torch.__version__, cuda=torch.version.cuda))
        event(root, stage='engine_ready', train=len(items['train']), validation=len(items['validation']))
        for split in ('train', 'validation'):
            profile_split(wrapper, args, items[split], root, split)
            export_dataset(root, items[split], split)
            event(root, stage='split_profile_complete', split=split, questions=len(items[split]))
        from scripts.verify_controller_pool import verify
        report = verify(root)
        write_json(root / 'verification.json', report)
        if not report['passed']:
            raise RuntimeError('Archive verification failed')
        write_json(root / 'finished.json', dict(status='complete', train=cli.train_samples,
            validation=cli.validation_samples, main_B_completions=frozen['expected_main_B_completions'],
            verified=True, training_started=False, historical_test300_used=False,
            total_instrumented_seconds=time.perf_counter() - started))
        event(root, stage='profiling_complete', **report)


if __name__ == '__main__':
    main()
