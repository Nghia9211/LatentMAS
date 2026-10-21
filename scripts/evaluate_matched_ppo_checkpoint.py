#!/usr/bin/env python3
"""Evaluate the latest committed early-stop checkpoint; never resume training.

Uses the frozen matched-run helpers/protocol, without fabricating a completed
200-iteration training marker. Writes results to a separate evaluation directory.
"""
import argparse
from datetime import datetime, timezone
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
from controller.profile import file_hash, write_json, write_tensors
from controller.matched import (test_items, render_batch, reference_prompt_audit,
                                CohortDecisions, online_rollout, prefixes_for_b)
from controller.checkpoints import load_controller_checkpoint
from latent_batch import rollout
from scripts.run_matched_controller_ppo import (completed, new_attempt, commit,
                                               reset_cache, decode_cohort, event)


def latest_checkpoint(training):
    """Only consecutive checksum-verified commits, never an unfinished attempt."""
    selected = None
    for n in range(1, 201):
        path = completed(training / 'train' / f'iteration{n:03d}')
        if path is None:
            break
        result = json.loads((path / 'result.json').read_text())
        if result['iteration'] != n or not result['train_only'] or result['episodes'] != 32:
            raise ValueError('Invalid committed training result')
        if not (path / 'checkpoint.pt').is_file():
            raise ValueError('Committed checkpoint missing')
        selected = (n, path / 'checkpoint.pt')
    if selected is None:
        raise ValueError('No completed training iteration to evaluate')
    return selected


@torch.no_grad()
def run_evaluation(wrapper, args, root, model, items, arithmetic, iteration):
    texts, encoded = render_batch(wrapper, args, items)
    reports = {}
    for mode in ('PPO', 'K10', 'K20', 'K80'):
        base = root / 'evaluation' / mode
        path = completed(base)
        if path is None:
            path = new_attempt(base)
            reset_cache(wrapper)
            torch.cuda.synchronize()
            started = time.perf_counter()
            if mode == 'PPO':
                decisions = CohortDecisions(model, 16, args.device, False)
                past, prefixes, physical = online_rollout(
                    wrapper, encoded['A']['input_ids'].to(args.device2),
                    encoded['A']['attention_mask'].to(args.device2), decisions,
                    preserve_arithmetic=arithmetic == 'preserve')
                budgets = decisions.budgets.cpu().tolist()
                write_tensors(path / 'controller_observations.safetensors',
                              dict(features=decisions.x, visited=decisions.visited))
            else:
                k = int(mode[1:])
                past, prefixes = rollout(wrapper,
                    encoded['A']['input_ids'].to(args.device2),
                    encoded['A']['attention_mask'].to(args.device2), latent_steps=k)
                budgets, physical = [k] * 16, k
            del past
            payloads = prefixes_for_b(wrapper, texts, encoded, prefixes)
            torch.cuda.synchronize()
            a_seconds = time.perf_counter() - started
            write_json(path / 'prompts.json', texts)
            rows, b_seconds = decode_cohort(wrapper, args, items, payloads,
                budgets, root, path, 'heldout_' + mode)
            summary = dict(mode=mode, checkpoint_iteration=iteration, samples=16,
                correct=sum(r['strict_correct'] for r in rows),
                strict_accuracy=sum(r['strict_correct'] for r in rows) / 16,
                legacy_correct=sum(r['legacy_correct'] for r in rows),
                parse_failures=sum(r['strict_parse_failed'] for r in rows),
                selected_budgets=budgets, mean_K=sum(budgets) / 16,
                cohort_physical_A_steps=physical,
                total_B_tokens=sum(r['output_tokens'] for r in rows),
                mean_B_tokens=sum(r['output_tokens'] for r in rows) / 16,
                caps=sum(r['finish_reason'] == 'length' for r in rows),
                loops=sum(r['repetition']['loop_flag'] for r in rows),
                A_controller_prefix_instrumented_seconds=a_seconds,
                B_cohort_seconds=b_seconds,
                instrumented_A_plus_B_seconds=a_seconds + b_seconds,
                timing_caveat='Excludes one-time model init/rendering; A phase includes observation serialization. Not production request latency.',
                test_used_for_training=False)
            write_json(path / 'summary.json', summary)
            commit(base, path)
            del payloads, prefixes
            event(root, stage='method_complete', **summary)
        reports[mode] = json.loads((path / 'summary.json').read_text())
    if not (root / 'evaluation_finished.json').exists():
        write_json(root / 'evaluation_finished.json', dict(status='complete',
            checkpoint_iteration=iteration, reports=reports,
            matched_first16=True, checkpoint_selection='Latest committed at researcher stop',
            test_used_for_training_or_checkpoint_selection=False,
            caveat='Previously exposed historical test16; smoke comparison, not pristine holdout.'))
    event(root, stage='evaluation_complete', reports=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args()
    training, root = cli.training.resolve(), cli.out.resolve()
    if root == ROOT or root == training or training in root.parents or len(root.parts) < 5:
        raise ValueError('Use a dedicated, separate evaluation directory')
    # Hold the existing training lock, preventing simultaneous resume/training.
    training_lock = (training / 'pipeline.lock').open('r')
    fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((training / 'manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    cfg = manifest['config']
    if 'reference_sha256' in manifest and file_hash(cfg['reference']) != manifest['reference_sha256']:
        raise ValueError('Historical reference changed')
    if 'test300_sha256' in manifest and file_hash(ROOT / 'data/medqa.json') != manifest['test300_sha256']:
        raise ValueError('Historical MedQA300 changed')
    iteration, checkpoint = latest_checkpoint(training)
    lock = (json.loads((training / 'training_lock.json').read_text())
            if (training/'training_lock.json').exists() else
            dict(arithmetic='preserve',source_ids=manifest['train_ids']))
    if lock['arithmetic'] != 'preserve':
        raise ValueError('This evaluation requires the approved preserve arithmetic')
    items = test_items(cfg['data_dir'], cfg['reference'])
    if any(i['source_split'] != 'test' or i['source_id'] in lock['source_ids'] for i in items):
        raise ValueError('Train/test firewall failed')
    frozen = dict(training=str(training), checkpoint=str(checkpoint),
        checkpoint_sha256=file_hash(checkpoint), checkpoint_iteration=iteration,
        planned_iterations=200, completed_training_episodes=iteration * 32,
        selection='Latest committed checkpoint at researcher-requested stop; no test-based selection',
        config=cfg, arithmetic=lock['arithmetic'],
        test_source_ids=[i['source_id'] for i in items],
        train_test_source_overlap=False, test_used_for_training=False,
        source_sha256=manifest['source_sha256'],
        evaluator_sha256=file_hash(Path(__file__)),
        modes=['PPO', 'K10', 'K20', 'K80'],
        cache='Reset before each method; one ordered batch16',
        controller_decoding='argmax', B_temperature=.6, B_batch=16,
        primary_metric='strict boxed A/B/C/D')
    if cli.dry_run:
        print(json.dumps(frozen, indent=2))
        return
    if root.exists():
        if not cli.resume or json.loads((root / 'manifest.json').read_text()) != frozen:
            raise ValueError('Use new output or unchanged --resume protocol')
    else:
        root.mkdir(parents=True)
        write_json(root / 'manifest.json', frozen)
    output_lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'evaluation_finished.json').exists():
        for mode in frozen['modes']:
            if completed(root / 'evaluation' / mode) is None:
                raise ValueError('Missing method commit')
        print('Evaluation already complete; checksums verified; no GPU initialization')
        return
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['iteration'] != iteration:
        raise ValueError('Checkpoint payload iteration mismatch')
    args = SimpleNamespace(**cfg)
    os.environ.pop('VLLM_BATCH_INVARIANT', None)
    from models import ModelWrapper
    from utils import set_seed
    set_seed(42)
    event(root, stage='initializing', checkpoint_iteration=iteration,
          checkpoint_sha256=frozen['checkpoint_sha256'])
    wrapper = ModelWrapper(args.model_name, torch.device(args.device), use_vllm=True, args=args)
    workers = wrapper.vllm_engine.collective_rpc('profile_inspect_weights')
    if any(w['head_dtype'] != 'torch.bfloat16' or w['embedding_dtype'] != 'torch.bfloat16'
           or w['quantization'] is not None or w['model_revision'] != args.model_revision for w in workers):
        raise ValueError('B dtype/revision mismatch')
    if wrapper.HF_model.config._commit_hash != args.model_revision or next(wrapper.HF_model.parameters()).dtype != torch.bfloat16:
        raise ValueError('A dtype/revision mismatch')
    audit = reference_prompt_audit(wrapper, args, args.data_dir, args.reference)
    if not (root / 'prompt_audit.json').exists():
        write_json(root / 'prompt_audit.json', audit)
    model = load_controller_checkpoint(checkpoint,args.device,
        expected_schema=cfg.get('controller_normalization'))
    model.eval()
    if file_hash(checkpoint) != frozen['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after selection')
    if not (root / 'frozen_controller.safetensors').exists():
        write_tensors(root / 'frozen_controller.safetensors', model.state_dict())
    write_json(root / f'preflight_{time.time_ns()}.json', dict(workers=workers,
        hf_dtype='torch.bfloat16', checkpoint_iteration=iteration,
        no_optimizer_created=True, train_source_unchanged=True,
        gpu=torch.cuda.get_device_name(), utc=datetime.now(timezone.utc).isoformat()))
    event(root, stage='engine_ready', checkpoint_iteration=iteration)
    try:
        run_evaluation(wrapper, args, root, model, items, lock['arithmetic'], iteration)
    except Exception as error:
        event(root, stage='failed', type=type(error).__name__, message=str(error))
        raise


if __name__ == '__main__':
    main()
