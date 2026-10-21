#!/usr/bin/env python3
"""Rotating train-pool PPO in a NEW run forked from an explicitly chosen pilot.

Default stage is CPU-only prepare. Training must be explicitly requested, with
an explicit iteration count. Never resumes/mutates the historical parent run.
Uses live rewards on selected train questions; the old train32 teacher remains a
separate fixed rehearsal auxiliary, not counterfactual labels for the new pool.
"""
import argparse
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
from controller.sampler import train_pool, ShuffledQuestionSampler
from controller.profile import file_hash, write_json, write_tensors, atomic_exclusive_bytes
from controller.matched import render_batch, CohortDecisions, online_rollout, prefixes_for_b
from controller.ppo import PPOController
from scripts.evaluate_matched_ppo_checkpoint import latest_checkpoint
from scripts.run_matched_controller_ppo import (
    SOURCES as BASE_SOURCES, completed, new_attempt, commit, event, reset_cache,
    decode_cohort, ppo_update, save_checkpoint, require_train)

SOURCES = list(BASE_SOURCES) + ['controller/sampler.py', 'scripts/run_pooled_controller_ppo.py',
                             'scripts/evaluate_matched_ppo_checkpoint.py']


def run_iteration(wrapper, args, root, items, model, optimizer, teacher_x, ties,
                  cfg, plan, parent_iteration, auxiliary_pool_size=32):
    """Unchanged rollout32 -> four PPO epochs -> sixteen optimizer updates."""
    require_train(items)
    if [i['source_id'] for i in items] != plan['source_ids']:
        raise ValueError('Question cohort differs from frozen sampler plan')
    local = plan['local_iteration']
    base = root / 'train' / f'iteration{local:03d}'
    path = new_attempt(base)
    write_json(path / 'sampler_plan.json', plan)
    torch.cuda.synchronize()
    started = time.perf_counter()
    reset_cache(wrapper)
    fields = {key: [] for key in ('features', 'actions', 'old_logprobs', 'old_values', 'visited')}
    rewards, budgets, physical = [], [], []
    # No optimizer step until BOTH cohorts have finished with the same pi_old.
    for group in range(2):
        cohort = items[group * 16:(group + 1) * 16]
        texts, encoded = render_batch(wrapper, args, cohort)
        decisions = CohortDecisions(model, 16, args.device, True)
        past, prefixes, steps = online_rollout(wrapper,
            encoded['A']['input_ids'].to(args.device2),
            encoded['A']['attention_mask'].to(args.device2), decisions,
            preserve_arithmetic=True)
        del past
        payloads = prefixes_for_b(wrapper, texts, encoded, prefixes)
        sub = path / f'cohort{group}'
        sub.mkdir()
        write_json(sub / 'prompts.json', texts)
        write_tensors(sub / 'observations.safetensors', dict(features=decisions.x,
            visited=decisions.visited,
            **{f'observed_h_{i}': torch.stack(h) for i, h in enumerate(decisions.histories)}))
        ks = decisions.budgets.cpu().tolist()
        rows, _ = decode_cohort(wrapper, args, cohort, payloads, ks, root, sub, 'live_train_reward')
        if len(rows) != 16 or [r['source_id'] for r in rows] != [i['source_id'] for i in cohort]:
            raise ValueError('Incomplete or reordered live training outcomes')
        rewards.extend(float(r['strict_correct']) for r in rows)
        budgets.extend(ks)
        physical.append(steps)
        for key, value in [('features', decisions.x), ('actions', decisions.actions),
                ('old_logprobs', decisions.logprobs), ('old_values', decisions.values),
                ('visited', decisions.visited)]:
            fields[key].append(value.detach())
        del payloads, prefixes, decisions
    data = {key: torch.cat(value) for key, value in fields.items()}
    reward = torch.tensor(rewards, device=args.device)
    episode = {key: data[key] for key in ('actions', 'old_logprobs', 'old_values', 'visited')}
    episode['returns'] = reward[:, None].expand(32, 7).clone()
    episode['advantages'] = (episode['returns'] - episode['old_values']) * episode['visited']
    write_tensors(path / 'episodes.safetensors', dict(**data, rewards=reward,
        returns=episode['returns'], advantages=episode['advantages']))
    # Separate TRAIN profile observations, not labels for live rollout states.
    # Default callers use old train32; the fresh-profile runner selects32 from256.
    loss, updates = ppo_update(model, optimizer, data['features'], episode, teacher_x, ties, cfg)
    if updates != 16:
        raise RuntimeError('Expected unchanged four epochs x four minibatches')
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    absolute = parent_iteration + local
    result = dict(iteration=local, total_controller_iterations=absolute,
        train_only=True, episodes=32, actual_B_completions=32, correct=int(reward.sum()),
        selected_budgets=budgets, mean_selected_K=sum(budgets) / 32,
        cohort_physical_steps=physical, loss=loss, optimizer_updates=updates,
        seconds_including_live_A_B=seconds, sampler_epoch=plan['epoch'],
        iteration_in_epoch=plan['iteration_in_epoch'], pool_size=plan['pool_size'],
        auxiliary_questions_this_iteration=len(teacher_x), auxiliary_pool_questions=auxiliary_pool_size,
        auxiliary_is_current_rollout_observation_labels=False,
        test_used=False, checkpoint_selected_by_test=False)
    if auxiliary_pool_size == 32:
        result.update(fixed_train_auxiliary_questions=32, auxiliary_is_current_question_labels=False)
    write_json(path / 'result.json', result)
    save_checkpoint(path / 'checkpoint.pt', model, optimizer, absolute)
    commit(base, path)
    event(root, stage='PPO_iteration_complete', iteration=absolute, local_iteration=local,
        correct=result['correct'], seconds=seconds, mean_K=result['mean_selected_K'],
        sampler_epoch=plan['epoch'], iteration_in_epoch=plan['iteration_in_epoch'],
        pool_size=plan['pool_size'])
    return path


def committed_progress(root, sampler, iterations, parent_iteration):
    """Only committed, checksum-verified attempts advance the sampler."""
    last, checkpoint = 0, None
    gap = False
    for local in range(1, iterations + 1):
        path = completed(root / 'train' / f'iteration{local:03d}')
        if path is None:
            gap = True
            continue
        if gap:
            raise ValueError('Non-consecutive committed iterations')
        if json.loads((path / 'sampler_plan.json').read_text()) != sampler.plan(local):
            raise ValueError('Committed sampler plan changed')
        result = json.loads((path / 'result.json').read_text())
        if (result['iteration'] != local or result['total_controller_iterations'] != parent_iteration + local
                or not result['train_only'] or result['episodes'] != 32):
            raise ValueError('Invalid committed pooled iteration')
        last, checkpoint = local, path / 'checkpoint.pt'
    return last, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True, help='Historical stopped matched train32 run')
    parser.add_argument('--checkpoint-iteration', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-samples', type=int, default=256)
    parser.add_argument('--sampler-seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, required=True, help='NEW local PPO iterations, not parent total')
    parser.add_argument('--stage', choices=('prepare', 'train'), default='prepare')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args()
    training, root = cli.training.resolve(), cli.out.resolve()
    if (root == ROOT or root == training or training in root.parents or root in training.parents
            or ROOT not in root.parents or cli.iterations < 1):
        raise ValueError('Use a dedicated new output under this repo, separate from parent; iterations >= 1')
    with (training / 'pipeline.lock').open('r') as parent_lock:
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent = json.loads((training / 'manifest.json').read_text())
        for name, digest in parent['source_sha256'].items():
            if file_hash(ROOT / name) != digest:
                raise ValueError('Frozen parent source changed: ' + name)
        parent_iteration, checkpoint = latest_checkpoint(training)
        if parent_iteration != cli.checkpoint_iteration:
            raise ValueError('Latest committed parent checkpoint differs from requested iteration')
        lock = json.loads((training / 'training_lock.json').read_text())
        cfg = parent['config']
        if (lock['arithmetic'] != 'preserve' or cfg['optimization_epochs'] != 4
                or cfg['minibatch_questions'] != 8 or cfg['generate_bs'] != 16
                or cfg['seed'] != 42 or cfg['time_penalty'] != 0):
            raise ValueError('Parent is not the approved PPO/protocol')
        warm_manifest = json.loads((training / 'warm_dataset_manifest.json').read_text())
        if (warm_manifest['split'] != 'train' or warm_manifest['source_ids'] != lock['source_ids']
                or len(lock['source_ids']) != 32
                or file_hash(training / 'warm_dataset.safetensors') != warm_manifest['file_sha256']):
            raise ValueError('Fixed train-only rehearsal provenance/hash mismatch')
        items, split = train_pool(cfg['data_dir'], cli.train_samples, cli.sampler_seed, lock['source_ids'])
        sampler = ShuffledQuestionSampler(split['source_ids'], cli.sampler_seed)
        frozen = dict(training=str(training), parent_iteration=parent_iteration,
            checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
            parent_config=cfg, parent_source_sha256=parent['source_sha256'],
            parent_train_ids=lock['source_ids'], train_manifest=split,
            warm_dataset_sha256=warm_manifest['file_sha256'], iterations=cli.iterations,
            sampler=dict(version='train-pool-v1', seed=cli.sampler_seed,
                episodes_per_iteration=32, cohort_sizes=[16, 16],
                iterations_per_epoch=sampler.iterations_per_epoch,
                shuffle_each_epoch=True, no_replacement_within_epoch=True),
            source_sha256={name: file_hash(ROOT / name) for name in SOURCES},
            expected_live_B_completions=32 * cli.iterations, new_full_grid_profile_required=False,
            auxiliary='Fixed old train32 rehearsal, separate from live pool states and rewards',
            test_used_for_training=False, parent_run_mutated=False)
        if root.exists():
            if not cli.resume:
                raise FileExistsError('Output exists; use --resume only for this NEW pooled run')
            if json.loads((root / 'manifest.json').read_text()) != frozen:
                raise ValueError('Resume source, pool, parent or protocol changed')
        elif cli.resume:
            raise ValueError('No prepared pooled run to resume')
        if cli.dry_run:
            print(json.dumps(dict(**frozen, first_sampler_plan=sampler.plan(1),
                next_epoch_plan=sampler.plan(sampler.iterations_per_epoch + 1)), indent=2))
            return
        if not root.exists():
            root.mkdir(parents=True)
            write_json(root / 'manifest.json', frozen)
            for name in SOURCES:
                atomic_exclusive_bytes(root / 'source_snapshot' / name, (ROOT / name).read_bytes())
        with (root / 'pipeline.lock').open('a') as output_lock:
            fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            last, local_checkpoint = committed_progress(root, sampler, cli.iterations, parent_iteration)
            event(root, stage='pool_prepared', train_samples=len(items), completed_local_iterations=last,
                iterations_per_epoch=sampler.iterations_per_epoch, test_used=False)
            if cli.stage == 'prepare':
                return
            if last == cli.iterations:
                # Recover a crash after the final commit but before the marker,
                # without loading LLMs or repeating any completed rollout.
                if not (root / 'training_finished.json').exists():
                    write_json(root / 'training_finished.json', dict(status='complete',
                        local_iterations=cli.iterations,
                        total_controller_iterations=parent_iteration + cli.iterations,
                        local_train_episodes=32 * cli.iterations, pool_size=len(items), test_used=False))
                return
            os.environ.pop('VLLM_BATCH_INVARIANT', None)
            from models import ModelWrapper
            from utils import set_seed
            args = SimpleNamespace(**cfg)
            set_seed(42)
            wrapper = ModelWrapper(args.model_name, torch.device(args.device), use_vllm=True, args=args)
            workers = wrapper.vllm_engine.collective_rpc('profile_inspect_weights')
            if any(w['head_dtype'] != 'torch.bfloat16' or w['embedding_dtype'] != 'torch.bfloat16'
                    or w['quantization'] is not None or w['model_revision'] != args.model_revision for w in workers):
                raise ValueError('B precision/revision mismatch')
            if (wrapper.HF_model.config._commit_hash != args.model_revision
                    or next(wrapper.HF_model.parameters()).dtype != torch.bfloat16):
                raise ValueError('A precision/revision mismatch')
            saved = torch.load(local_checkpoint or checkpoint, map_location=args.device, weights_only=True)
            if saved['iteration'] != parent_iteration + last:
                raise ValueError('Checkpoint iteration mismatch')
            model = PPOController(7685, cfg['width']).to(args.device)
            model.load_state_dict(saved['model'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=0)
            optimizer.load_state_dict(saved['optimizer'])
            teacher = load_file(str(training / 'warm_dataset.safetensors'))
            teacher_x = teacher['features'].to(args.device)
            ties = (teacher['q_targets'][:, :, 0] == teacher['q_targets'][:, :, 1]).to(args.device)
            torch.set_rng_state(saved['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
            for local in range(last + 1, cli.iterations + 1):
                plan = sampler.plan(local)
                batch = [items[i] for i in plan['indices']]
                run_iteration(wrapper, args, root, batch, model, optimizer, teacher_x, ties,
                              cfg, plan, parent_iteration)
            write_json(root / 'training_finished.json', dict(status='complete',
                local_iterations=cli.iterations, total_controller_iterations=parent_iteration + cli.iterations,
                local_train_episodes=32 * cli.iterations, pool_size=len(items), test_used=False))


if __name__ == '__main__':
    main()
