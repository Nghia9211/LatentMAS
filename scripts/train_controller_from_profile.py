#!/usr/bin/env python3
"""Verified train256 fresh warm-start ->80 live PPO iterations ->val every16.

Requires a supervisor release proving three hours of GPU rest. Never loads test
prompts or old controller weights. Fixed baselines are read from the matching
verified validation profile, not generated again. All outputs are a NEW run.
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
from controller.job_gate import check_release, profile_ready
from controller.profile import file_hash, write_json, write_tensors, atomic_exclusive_bytes, load_official_medqa
from controller.matched import historical_style, render_batch, CohortDecisions, online_rollout, prefixes_for_b
from controller.sampler import ShuffledQuestionSampler
from controller.warmstart import WarmStartController, check_gradients
from controller.ppo import PPOController
from controller.training import logged_ppo_update
from controller.tracking import TrainingTracker, require_wandb
from controller.metrics import rollout_metrics, outcome_metrics, read_outcomes, prior_coverage
from scripts.run_matched_controller_ppo import completed, new_attempt, commit, reset_cache, decode_cohort, save_checkpoint, event
from scripts.validate_controller_wandb import preserve_rng_and_mode
from controller.training import train_update
from scripts import run_pooled_controller_ppo as core

VALIDATIONS = (16, 32, 48, 64, 80)
EXTRA_SOURCES = ['controller/job_gate.py', 'controller/tracking.py', 'controller/metrics.py',
    'scripts/run_pooled_controller_ppo.py',
    'scripts/train_controller_from_profile.py', 'controller/training.py']


def profile_items(profile, manifest, split, data_root=None):
    """Return only exact approved identities from ONE declared official split."""
    if split not in ('train', 'validation'):
        raise ValueError('Test examples are forbidden')
    pool, metadata = load_official_medqa(data_root if data_root is not None else manifest['config']['data_dir'], split)
    selection = manifest['selections'][split]
    if metadata['sha256'] != selection['dataset_sha256']:
        raise ValueError('Dataset changed')
    lookup = {i['source_id']: i for i in pool}
    ids = selection['source_ids']
    if len(ids) != (256 if split == 'train' else 128) or len(set(ids)) != len(ids):
        raise ValueError('Expected exact train256 / validation128 identities')
    items = [dict(lookup[i], question=historical_style(lookup[i])) for i in ids]
    if any(i['source_split'] != split or f':{split}:' not in i['source_id'] for i in items):
        raise ValueError('Split firewall failed')
    return items


def training_tensors(profile, source_ids):
    path = completed(Path(profile) / 'train/dataset')
    if path is None:
        raise ValueError('No committed train dataset')
    meta = json.loads((path / 'manifest.json').read_text())
    if meta['split'] != 'train' or meta['source_ids'] != source_ids or meta['allowed_for_gradient_updates'] is not True:
        raise ValueError('Only matching official train profiles can enter optimizer')
    d = load_file(str(path / 'dataset.safetensors'))
    if (d['features'].shape != (256, 7, 7685) or d['actions'].shape != (256, 7)
            or d['q_targets'].shape != (256, 7, 2) or d['correctness'].shape != (256, 8)
            or not all(torch.isfinite(t).all() for t in d.values())):
        raise ValueError('Unexpected full train dataset shape/finiteness')
    return d


def warmstart(model, optimizer, d, cfg, root, tracker):
    """Same50 full-batch warm-start epochs, now all256 train questions."""
    base = root / 'warmstart'
    path = completed(base)
    if path is None:
        path = new_attempt(base)
        warm = WarmStartController('raw_mlp', 7685, 128).to(cfg['device'])
        warm_opt = torch.optim.AdamW(warm.parameters(), lr=cfg['learning_rate'], weight_decay=0)
        started = time.perf_counter()
        losses = []
        for epoch in range(1, cfg['warmstart_epochs'] + 1):
            loss, norm = train_update(warm, warm_opt, d['features'], d['actions'], d['q_targets'],
                dict(q_weight=1., grad_clip=cfg['grad_clip']))
            losses.append(loss)
            tracker.log({'warmstart/epoch': epoch, 'warmstart/loss': loss, 'warmstart/gradient_norm': norm})
        model.load_warmstart(warm)
        if optimizer.state:
            raise ValueError('Fresh PPO optimizer must be empty after warm-start')
        torch.cuda.synchronize()
        save_checkpoint(path / 'checkpoint.pt', model, optimizer, 0)
        write_json(path / 'result.json', dict(epochs=cfg['warmstart_epochs'], train_questions=256,
            optimizer_updates=cfg['warmstart_epochs'], losses=losses, seconds=time.perf_counter() - started,
            old_checkpoint_loaded=False, validation_used_for_gradient=False,
            PPO_critic_new_bias=.5, PPO_optimizer_fresh=True))
        commit(base, path)
        del warm, warm_opt
    return path / 'checkpoint.pt'


@torch.no_grad()
def validation(wrapper, args, model, items, root, iteration, baselines, baseline_source='Verified profile validation grid'):
    """128 actual PPO completions only; RNG/mode restored even on failure."""
    if len(items) != 128 or any(i['source_split'] != 'validation' for i in items):
        raise ValueError('Exact validation-only128 cohort required')
    base = root / 'validation' / f'iteration{iteration:03d}'
    existing = completed(base)
    if existing is not None:
        return json.loads((existing / 'metrics.json').read_text())
    path = new_attempt(base)
    rows = []
    started = time.perf_counter()
    with preserve_rng_and_mode(model):
        reset_cache(wrapper)
        for group in range(8):
            cohort = items[group * 16:(group + 1) * 16]
            texts, encoded = render_batch(wrapper, args, cohort)
            decisions = CohortDecisions(model, 16, args.device, False)
            past, prefixes, physical = online_rollout(wrapper,
                encoded['A']['input_ids'].to(args.device2), encoded['A']['attention_mask'].to(args.device2),
                decisions, preserve_arithmetic=True)
            del past
            payloads = prefixes_for_b(wrapper, texts, encoded, prefixes)
            sub = path / f'cohort{group}'
            sub.mkdir()
            write_json(sub / 'prompts.json', texts)
            write_tensors(sub / 'observations.safetensors', dict(features=decisions.x, visited=decisions.visited))
            budgets = decisions.budgets.cpu().tolist()
            decoded, _ = decode_cohort(wrapper, args, cohort, payloads, budgets, root, sub, 'validation_PPO')
            if [r['source_id'] for r in decoded] != [i['source_id'] for i in cohort]:
                raise ValueError('Validation cohort reordered/incomplete')
            rows.extend(decoded)
            del payloads, prefixes, decisions
    metrics = {'validation/available': 1, 'validation/samples': 128,
        'validation/PPO/accuracy': sum(r['strict_correct'] for r in rows) / 128,
        'validation/PPO/mean_K': sum(r['k'] for r in rows) / 128,
        'validation/PPO/B_tokens_mean': sum(r['output_tokens'] for r in rows) / 128,
        'validation/PPO/cap_rate': sum(r['finish_reason'] == 'length' for r in rows) / 128,
        'validation/PPO/loop_rate': sum(r['repetition']['loop_flag'] for r in rows) / 128,
        'validation/PPO/parse_failure_rate': sum(r['strict_parse_failed'] for r in rows) / 128,
        'validation/report_only_seconds': time.perf_counter() - started}
    for k, baseline in baselines.items():
        metrics[f'validation/fixed_K{k}/accuracy'] = baseline['accuracy']
        metrics[f'validation/PPO_minus_K{k}_accuracy'] = metrics['validation/PPO/accuracy'] - baseline['accuracy']
    write_json(path / 'metrics.json', metrics)
    write_json(path / 'provenance.json', dict(iteration=iteration, source_ids=[i['source_id'] for i in items],
        actual_B_completions=128, fixed_baseline_calls=0, fixed_baselines_source=baseline_source,
        cache_protocol='cold_sequence_start_warm_between_eight_ordered_cohorts',
        labels_used_for_gradient=False, test300_loaded=False,
        caveat='Same locked protocol, but fixed results and current PPO results are from different engine lifetimes; no bitwise stochastic-repeatability claim.'))
    commit(base, path)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--wandb-mode', choices=('disabled', 'offline', 'online'), default='online')
    p.add_argument('--wandb-project', default='Controller-Latent-Reasoning')
    p.add_argument('--wandb-entity', default='theanha2hvtls-ai4life')
    p.add_argument('--wandb-name', default='ppo-train256-freshwarm-80iters')
    cli = p.parse_args()
    profile, root = cli.profile.resolve(), cli.out.resolve()
    if (ROOT not in root.parents or root == profile or root in profile.parents or profile in root.parents):
        raise ValueError('Use a dedicated separate training output')
    m = json.loads((profile / 'manifest.json').read_text())
    cfg = dict(m['config'], train_samples=256, validation_samples=128, iterations=80, warmstart_epochs=50)
    if (cfg['width'] != 128 or cfg['temperature'] != .6 or cfg['generate_bs'] != 16
            or cfg['optimization_epochs'] != 4 or cfg['minibatch_questions'] != 8 or cfg['time_penalty'] != 0):
        raise ValueError('Not the approved training protocol')
    names = list(dict.fromkeys(list(m['source_sha256']) + EXTRA_SOURCES))
    sources = {name: file_hash(ROOT / name) for name in names}
    for name, digest in m['source_sha256'].items():
        if sources[name] != digest:
            raise ValueError('Frozen profile source changed: ' + name)
    planned = dict(config=cfg, profile=str(profile), profile_manifest_sha256=file_hash(profile / 'manifest.json'),
        source_sha256=sources, train_ids=m['selections']['train']['source_ids'],
        validation_ids=m['selections']['validation']['source_ids'],
        warmstart='Fresh random MLP128 +50 full-batch epochs over verified train256; no old weights',
        auxiliary='Matched32 profile observations per iteration from the full train256 pool, separate from live states',
        PPO_iterations=80, expected_train_B_completions=2560, expected_validation_B_completions=640,
        validation_checkpoints=list(VALIDATIONS), fixed_baselines_reused=True, test300_used=False,
        wandb=dict(mode=cli.wandb_mode, project=cli.wandb_project, entity=cli.wandb_entity, name=cli.wandb_name))
    if cli.dry_run:
        print(json.dumps(planned, indent=2))
        return
    release = check_release(cli.release, profile, root)
    # Locks are acquired BEFORE any W&B init, warm-start or GPU initialization.
    with (profile / 'pipeline.lock').open('r') as profile_lock:
        fcntl.flock(profile_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from scripts.verify_controller_pool import verify
        report = verify(profile)
        if not report['passed']:
            raise ValueError('Independent profile verification did not pass')
        items = profile_items(profile, m, 'train')
        valid_items = profile_items(profile, m, 'validation')
        source_ids = [i['source_id'] for i in items]
        d = training_tensors(profile, source_ids)
        sampler = ShuffledQuestionSampler(source_ids, 42)
        planned.update(profile_completion_sha256=release['profile_sha256'], release_sha256=file_hash(cli.release))
        if root.exists():
            if not cli.resume or json.loads((root / 'manifest.json').read_text()) != planned:
                raise ValueError('Training output exists or resume manifest changed')
        else:
            if cli.resume:
                raise ValueError('No training run to resume')
            root.mkdir(parents=True)
            write_json(root / 'manifest.json', planned)
            for name in names:
                atomic_exclusive_bytes(root / 'source_snapshot' / name, (ROOT / name).read_bytes())
        with (root / 'pipeline.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            last, checkpoint = core.committed_progress(root, sampler, 80, 0)
            if (root / 'training_finished.json').exists():
                return
            require_wandb(cli.wandb_mode)
            tracker = TrainingTracker(root, mode=cli.wandb_mode, project=cli.wandb_project,
                entity=cli.wandb_entity, name=cli.wandb_name, config=dict(cfg, runner='profile_fresh'))
            if tracker.run is not None:
                tracker.run.define_metric('warmstart/epoch')
                tracker.run.define_metric('warmstart/*', step_metric='warmstart/epoch')
            code = 1
            try:
                os.environ.pop('VLLM_BATCH_INVARIANT', None)
                from models import ModelWrapper
                from utils import set_seed
                set_seed(42)
                args = SimpleNamespace(**cfg)
                model = PPOController(7685, 128).to(args.device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=0)
                d = {k: v.to(args.device) for k, v in d.items()}
                initial = warmstart(model, optimizer, d, cfg, root, tracker)
                saved = torch.load(checkpoint or initial, map_location=args.device, weights_only=True)
                model.load_state_dict(saved['model'])
                optimizer.load_state_dict(saved['optimizer'])
                if saved['iteration'] != last:
                    raise ValueError('Resume checkpoint iteration mismatch')
                if last == 0 and optimizer.state:
                    raise ValueError('Fresh PPO optimizer is not empty')
                wrapper = ModelWrapper(args.model_name, torch.device(args.device), use_vllm=True, args=args)
                workers = wrapper.vllm_engine.collective_rpc('profile_inspect_weights')
                if any(w['head_dtype'] != 'torch.bfloat16' or w['embedding_dtype'] != 'torch.bfloat16'
                        or w['quantization'] is not None or w['model_revision'] != args.model_revision for w in workers):
                    raise ValueError('B precision/revision mismatch')
                if (wrapper.HF_model.config._commit_hash != args.model_revision
                        or next(wrapper.HF_model.parameters()).dtype != torch.bfloat16):
                    raise ValueError('A precision/revision mismatch')
                torch.set_rng_state(saved['torch_rng'].cpu())
                torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
                seen, episodes_seen = prior_coverage(root, last)
                baselines = report['summaries']['validation']
                def evaluate(n):
                    metrics = validation(wrapper, args, model, valid_items, root, n, baselines)
                    tracker.log(dict(metrics, iteration=n))
                    event(root, stage='validation_complete', iteration=n,
                        accuracy=metrics['validation/PPO/accuracy'], mean_K=metrics['validation/PPO/mean_K'])
                # A committed checkpoint followed by an interrupted validation must
                # finish that validation before collecting any next train rollout.
                for n in VALIDATIONS:
                    if n < last and completed(root / 'validation' / f'iteration{n:03d}') is None:
                        raise ValueError('Missing earlier validation before later committed PPO iterations')
                if last in VALIDATIONS:
                    evaluate(last)
                ties = d['q_targets'][:, :, 0] == d['q_targets'][:, :, 1]
                original_update = core.ppo_update
                try:
                    for n in range(last + 1, 81):
                        plan = sampler.plan(n)
                        index = torch.tensor(plan['indices'], device=args.device)
                        batch = [items[i] for i in plan['indices']]
                        def update(model, opt, x, episode, teacher, masks, config):
                            def log(step, values):
                                tracker.log(dict(values, optimizer_step=(n - 1) * 16 + step, iteration=n))
                            started = time.perf_counter()
                            result = logged_ppo_update(model, opt, x, episode, teacher, masks, config, log)
                            torch.cuda.synchronize()
                            update.seconds = time.perf_counter() - started
                            return result
                        core.ppo_update = update
                        event_count = len((root / 'events.jsonl').read_text().splitlines()) if (root / 'events.jsonl').exists() else 0
                        path = core.run_iteration(wrapper, args, root, batch, model, optimizer,
                            d['features'][index], ties[index], cfg, plan, 0, auxiliary_pool_size=256)
                        result = json.loads((path / 'result.json').read_text())
                        episode = load_file(str(path / 'episodes.safetensors'))
                        outcomes = read_outcomes(path)
                        seen.update(i['source_id'] for i in outcomes)
                        episodes_seen += 32
                        new_events = [json.loads(s) for s in (root / 'events.jsonl').read_text().splitlines()[event_count:]]
                        b_seconds = sum(e['seconds'] for e in new_events if e.get('stage') == 'B_cohort_complete')
                        metrics = dict(rollout_metrics(episode), **outcome_metrics(outcomes))
                        metrics.update({'iteration/loss_total': result['loss'], 'iteration/committed': 1,
                            'train/accuracy': result['correct'] / 32, 'train/mean_K': result['mean_selected_K'],
                            'time/iteration_seconds': result['seconds_including_live_A_B'],
                            'time/B_generate_seconds': b_seconds, 'time/PPO_update_seconds': update.seconds,
                            'budget/cohort_physical_steps_max': max(result['cohort_physical_steps']),
                            'data/episodes_seen': episodes_seen, 'data/unique_PPO_questions_seen': len(seen),
                            'data/unique_auxiliary_questions': 256, 'data/train_pool_size': 256,
                            'data/sampler_epoch': plan['epoch'], 'data/iteration_in_epoch': plan['iteration_in_epoch'],
                            'validation/available': 0})
                        tracker.log(dict(metrics, iteration=n))
                        if n in VALIDATIONS:
                            evaluate(n)
                finally:
                    core.ppo_update = original_update
                validations = []
                for n in VALIDATIONS:
                    path = completed(root / 'validation' / f'iteration{n:03d}')
                    values = json.loads((path / 'metrics.json').read_text())
                    validations.append(dict(iteration=n, accuracy=values['validation/PPO/accuracy'], mean_K=values['validation/PPO/mean_K']))
                best = min(validations, key=lambda r: (-r['accuracy'], r['mean_K'], r['iteration']))
                write_json(root / 'training_finished.json', dict(status='complete', iterations=80,
                    train_episodes=2560, validation_B_completions=640, validation_checkpoints=list(VALIDATIONS),
                    validations=validations, best_validation=best,
                    checkpoint_selection='Max validation accuracy, then smaller meanK, then earlier iteration; no test/time selection',
                    old_controller_weights_loaded=False, test300_used=False,
                    fixed_baseline_generations=0, model_weights_frozen=True))
                event(root, stage='training_complete', iterations=80)
                code = 0
            finally:
                tracker.finish(code)


if __name__ == '__main__':
    main()
