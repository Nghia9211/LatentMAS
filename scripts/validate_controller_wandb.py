#!/usr/bin/env python3
"""Explicit official-validation evaluation, never training or test300 evaluation.

Runs PPO/fixedK10/K20/K80 only when invoked without --dry-run. Preserves RNG and
controller parameters; uploads scalar summaries only if --wandb-mode online.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from controller.profile import (load_official_medqa, normalized_question, make_subset,
                                write_json, file_hash)
from controller.matched import (historical_style, render_batch, online_rollout,
                                prefixes_for_b, CohortDecisions)
from controller.ppo import PPOController
from controller.tracking import TrainingTracker, require_wandb
from latent_batch import rollout
from scripts.run_matched_controller_ppo import reset_cache, decode_cohort
from scripts.evaluate_matched_ppo_checkpoint import latest_checkpoint


def validation_items(data_dir, count=128, seed=42):
    if count < 16 or count % 16:
        raise ValueError('Validation count must be a positive multiple of16, at least16')
    pool, metadata = load_official_medqa(data_dir, 'validation')
    train, _ = load_official_medqa(data_dir, 'train')
    forbidden = {normalized_question(i['question_stem']) for i in train}
    pool = [i for i in pool if normalized_question(i['question_stem']) not in forbidden]
    items = [dict(i, question=historical_style(i)) for i in make_subset(pool, count, seed)]
    if any(i['source_split'] != 'validation' or ':validation:' not in i['source_id'] for i in items):
        raise ValueError('Only official validation examples allowed')
    return items, dict(split='validation', count=count, seed=seed, dataset_sha256=metadata['sha256'],
                       source_ids=[i['source_id'] for i in items], train_question_overlap=0,
                       test300_loaded=False, selection='Fixed seed, not outcomes')


@contextmanager
def preserve_rng_and_mode(model):
    cpu_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    training = model.training
    try:
        model.eval()
        yield
    finally:
        model.train(training)
        torch.set_rng_state(cpu_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


@torch.no_grad()
def evaluate_validation(wrapper, args, model, items, root, iteration):
    if len(items) % 16 or not items or any(i['source_split'] != 'validation' for i in items):
        raise ValueError('Validation-only ordered full cohorts16 required')
    metrics = dict(iteration=iteration, **{'validation/available': 1, 'validation/samples': len(items)})
    with preserve_rng_and_mode(model):
        for mode in ('PPO', 'K10', 'K20', 'K80'):
            reset_cache(wrapper)  # Cold each method, warm between its cohorts.
            rows = []
            for group in range(len(items) // 16):
                cohort = items[group * 16:(group + 1) * 16]
                texts, encoded = render_batch(wrapper, args, cohort)
                if mode == 'PPO':
                    decisions = CohortDecisions(model, 16, args.device, False)
                    past, prefixes, _ = online_rollout(wrapper,
                        encoded['A']['input_ids'].to(args.device2),
                        encoded['A']['attention_mask'].to(args.device2), decisions,
                        preserve_arithmetic=True)
                    budgets = decisions.budgets.cpu().tolist()
                else:
                    k = int(mode[1:])
                    past, prefixes = rollout(wrapper, encoded['A']['input_ids'].to(args.device2),
                        encoded['A']['attention_mask'].to(args.device2), latent_steps=k)
                    budgets = [k] * 16
                del past
                payloads = prefixes_for_b(wrapper, texts, encoded, prefixes)
                path = root / mode / f'cohort{group}'
                path.mkdir(parents=True)
                decoded, _ = decode_cohort(wrapper, args, cohort, payloads, budgets,
                                           root, path, 'validation_' + mode)
                rows.extend(decoded)
                del payloads, prefixes
            count = len(rows)
            metrics.update({f'validation/{mode}/accuracy': sum(r['strict_correct'] for r in rows) / count,
                f'validation/{mode}/mean_K': sum(r['k'] for r in rows) / count,
                f'validation/{mode}/B_tokens_mean': sum(r['output_tokens'] for r in rows) / count,
                f'validation/{mode}/cap_rate': sum(r['finish_reason'] == 'length' for r in rows) / count,
                f'validation/{mode}/parse_failure_rate': sum(r['strict_parse_failed'] for r in rows) / count,
                f'validation/{mode}/loop_rate': sum(r['repetition']['loop_flag'] for r in rows) / count})
        for fixed in ('K10', 'K20', 'K80'):
            metrics[f'validation/PPO_minus_{fixed}_accuracy'] = (
                metrics['validation/PPO/accuracy'] - metrics[f'validation/{fixed}/accuracy'])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--wandb-mode', choices=('disabled', 'offline', 'online'), default='offline')
    parser.add_argument('--wandb-project', default='Controller-Latent-Reasoning')
    parser.add_argument('--wandb-entity', default='theanha2hvtls-ai4life')
    parser.add_argument('--wandb-name')
    parser.add_argument('--wandb-run-id')
    cli = parser.parse_args()
    training, output = cli.training.resolve(), cli.out.resolve()
    if output.exists() or output == ROOT or output == training or training in output.parents:
        raise ValueError('Use a new, separate validation directory')
    handle = (training / 'pipeline.lock').open('r')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((training / 'manifest.json').read_text())
    for name, sha in manifest['source_sha256'].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError('Training source changed: ' + name)
    cfg = manifest['config']
    if json.loads((training / 'training_lock.json').read_text())['arithmetic'] != 'preserve':
        raise ValueError('Validation requires approved preserve arithmetic')
    iteration, checkpoint = latest_checkpoint(training)
    items, selection = validation_items(cfg['data_dir'], cli.samples, cli.seed)
    frozen = dict(iteration=iteration, checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
        config=cfg, selection=selection, expected_B_completions=4 * len(items),
        test300_used=False, training_run=False, source_sha256=file_hash(Path(__file__)))
    if cli.dry_run:
        print(json.dumps(frozen, indent=2))
        return
    require_wandb(cli.wandb_mode)
    output.mkdir(parents=True)
    write_json(output / 'manifest.json', frozen)
    # Establish logging/auth before spending GPU time on validation.
    tracker = TrainingTracker(output, mode=cli.wandb_mode, project=cli.wandb_project,
        entity=cli.wandb_entity, name=cli.wandb_name,
        config=dict(cfg, validation_samples=len(items), validation_seed=cli.seed), run_id=cli.wandb_run_id)
    status = 1
    try:
        from models import ModelWrapper
        from utils import set_seed
        os.environ.pop('VLLM_BATCH_INVARIANT', None)
        set_seed(cfg['seed'])
        args = SimpleNamespace(**cfg)
        wrapper = ModelWrapper(args.model_name, torch.device(args.device), use_vllm=True, args=args)
        workers = wrapper.vllm_engine.collective_rpc('profile_inspect_weights')
        if any(w['head_dtype'] != 'torch.bfloat16' or w['embedding_dtype'] != 'torch.bfloat16'
               or w['model_revision'] != args.model_revision or w['quantization'] is not None for w in workers):
            raise ValueError('B precision/revision differs from training')
        if wrapper.HF_model.config._commit_hash != args.model_revision or next(wrapper.HF_model.parameters()).dtype != torch.bfloat16:
            raise ValueError('A precision/revision differs from training')
        model = PPOController(7685, cfg['width']).to(args.device)
        saved = torch.load(checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(saved['model'])
        metrics = evaluate_validation(wrapper, args, model, items, output, iteration)
        if file_hash(checkpoint) != frozen['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during validation')
        write_json(output / 'validation_metrics.json', metrics)
        tracker.log(metrics)
        status = 0
    finally:
        tracker.finish(status)
    print(json.dumps(dict(status='complete', metrics=metrics, training_run=False), indent=2))


if __name__ == '__main__':
    main()
