"""Shared fresh-rollout PPO runtime for three-group observation schemas.

Only the controller is optimized. LLMs, sampling, reward and PPO update equations
are unchanged; resource admission happens before allocating either LLM.
"""
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from controller.normalized import encoder_diagnostics
from controller.profile import write_json
from controller.sampler import ShuffledQuestionSampler
from controller.tracking import TrainingTracker
from controller.metrics import rollout_metrics, outcome_metrics, read_outcomes, prior_coverage
from controller.training import logged_ppo_update
from scripts.run_matched_controller_ppo import completed, new_attempt, commit, save_checkpoint, event
from scripts.train_controller_from_profile import validation
from scripts import run_pooled_controller_ppo as core

ROOT = Path(__file__).resolve().parents[1]


def _model_types(cfg):
    from controller import simple_state, cosine_state

    for module in (simple_state, cosine_state):
        if cfg["controller_normalization"] == module.SCHEMA:
            prefix = "Simple" if module is simple_state else "Cosine"
            return getattr(module, prefix + "WarmStartController"), getattr(module, prefix + "PPOController")
    raise ValueError("Unknown three-group observation schema")


def _wait_for_model_memory(args, root, timeout_seconds=1800):
    """Recheck logical devices after tracking/init, immediately before LLM load."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        free_b, total_b = torch.cuda.mem_get_info(0)
        free_a, _ = torch.cuda.mem_get_info(torch.device(args.device2))
        need_b = max(16 * 1024**3, int(total_b * args.gpu_memory_utilization) + 1024**3)
        need_a = 20 * 1024**3
        if free_b >= need_b and free_a >= need_a:
            event(root, stage="model_memory_admitted", A_free_bytes=free_a, B_free_bytes=free_b)
            return
        event(root, stage="model_memory_wait", A_free_bytes=free_a, B_free_bytes=free_b)
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU memory changed before model allocation; no rollout collected")
        time.sleep(15)


def initialize_without_warmstart(model, optimizer, cfg, root, tracker):
    """Match the baseline's pre-optimization random draws; use no labels/steps.

    The supervised-controller class only supplies the same random encoder/actor
    initialization. No forward, loss, optimizer or supervised update is run.
    """
    if optimizer.state:
        raise ValueError('Random initialization requires a fresh PPO optimizer')
    base = root / 'initialization'
    path = completed(base)
    if path is None:
        path = new_attempt(base)
        prototype = _model_types(cfg)[0](2562, cfg['width'], eps=cfg['normalization_eps'],
            projection_norm=cfg['projection_norm']).to(cfg['device'])
        prototype.normalizer.load_state_dict(model.normalizer.state_dict())
        model.load_warmstart(prototype)
        save_checkpoint(path / 'checkpoint.pt', model, optimizer, 0)
        write_json(path / 'result.json', dict(initialization='matched_random_encoder_actor',
            warmstart_skipped=True, epochs=0, optimizer_updates=0, labels_used=False,
            old_checkpoint_loaded=False, PPO_optimizer_fresh=True, PPO_critic_new_bias=.5,
            normalization=cfg["controller_normalization"], projection_norm=cfg['projection_norm']))
        commit(base, path)
        tracker.log({'initialization/warmstart_skipped': 1, 'initialization/optimizer_updates': 0})
    return path / 'checkpoint.pt'


def stopping_probabilities(logits):
    """Exact budget probabilities along fixed causal profile trajectories."""
    if logits.ndim != 3 or logits.shape[1:] != (7, 2):
        raise ValueError('Expected seven binary decision checkpoints')
    p = logits.softmax(-1)
    survival = p[:, :, 1].cumprod(1)
    before = torch.cat((torch.ones_like(survival[:, :1]), survival[:, :-1]), dim=1)
    return torch.cat((before * p[:, :, 0], survival[:, -1:]), dim=1)


def warmstart(model, optimizer, d, cfg, root, tracker):
    if cfg["warmstart_epochs"] != 0:
        raise ValueError("Three-group runtime supports fresh PPO without warm-start")
    return initialize_without_warmstart(model, optimizer, cfg, root, tracker)


def train(root, manifest, d, normalizer, report, *, train_items=None, validation_items=None):
    cfg = manifest['config']
    if (train_items is None) != (validation_items is None):
        raise ValueError('Supply both declared train and validation items')
    if train_items is None:
        profile_manifest = json.loads((Path(manifest['profile']) / 'manifest.json').read_text())
        items = profile_items(manifest['profile'], profile_manifest, 'train', cfg['data_dir'])
        valid_items = profile_items(manifest['profile'], profile_manifest, 'validation', cfg['data_dir'])
    else:
        items, valid_items = train_items, validation_items
    pool_size = len(manifest['train_ids'])
    if ([i['source_id'] for i in items] != manifest['train_ids']
            or [i['source_id'] for i in valid_items] != manifest['validation_ids']
            or len(d['features']) != pool_size or cfg['train_samples'] != pool_size):
        raise ValueError('Training pool/tensors/manifest identities differ')
    sampler = ShuffledQuestionSampler(manifest['train_ids'], cfg['seed'])
    last, checkpoint = core.committed_progress(root, sampler, cfg['iterations'], 0)
    if (root / 'training_finished.json').exists():
        print('Training already complete; no GPU or W&B initialization')
        return
    tracker = TrainingTracker(root, mode=manifest['wandb']['mode'],
        project=manifest['wandb']['project'], entity=manifest['wandb']['entity'],
        name=manifest['wandb']['name'], config=dict(cfg, runner='normalized_profile_fresh'))
    if tracker.run is not None:
        tracker.run.define_metric('warmstart/epoch')
        tracker.run.define_metric('warmstart/*', step_metric='warmstart/epoch')
        tracker.run.config.update({'controller_normalization': cfg["controller_normalization"],
            'projection_norm': cfg['projection_norm'], 'normalization_eps': cfg['normalization_eps']})
    code = 1
    try:
        os.environ.pop('VLLM_BATCH_INVARIANT', None)
        from models import ModelWrapper
        from utils import set_seed
        set_seed(cfg['seed'])
        args = SimpleNamespace(**cfg)
        model = _model_types(cfg)[1](2562, cfg['width'], eps=cfg['normalization_eps'],
            projection_norm=cfg['projection_norm']).to(args.device)
        model.normalizer.load_state_dict(normalizer.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=0)
        d = {k: v.to(args.device) for k, v in d.items()}
        initial = warmstart(model, optimizer, d, cfg, root, tracker)
        saved = torch.load(checkpoint or initial, map_location=args.device, weights_only=True)
        if saved['iteration'] != last:
            raise ValueError('Resume checkpoint iteration mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        from controller.ppo import greedy_budgets
        with torch.no_grad():
            initial_budgets = greedy_budgets(model, d['features'])
            logits, _ = model(d['features'])
            probabilities = stopping_probabilities(logits)
            initial = dict(iteration=last, training_profile_only=True,
                mean_greedy_K=float(initial_budgets.float().mean()),
                greedy_K_counts={str(k):int((initial_budgets==k).sum()) for k in range(10,81,10)},
                mean_handoff_probability_by_checkpoint=logits.softmax(-1)[:,:,0].mean(0).tolist(),
                initialization='random_no_warmstart' if cfg['warmstart_epochs'] == 0 else 'supervised_warmstart',
                mean_stochastic_K=float((probabilities * torch.arange(10,81,10,device=probabilities.device)).sum(1).mean()),
                expected_stochastic_K_counts={str(k):float(probabilities[:,j].sum()) for j,k in enumerate(range(10,81,10))},
                encoder=encoder_diagnostics(model,d['features']))
        write_json(root / f'initial_policy_{time.time_ns()}.json', initial)
        event(root, stage='initial_policy_sanity', iteration=last,
              mean_greedy_K=initial['mean_greedy_K'])
        _wait_for_model_memory(args, root)
        wrapper = ModelWrapper(args.model_name, torch.device(args.device), use_vllm=True, args=args)
        workers = wrapper.vllm_engine.collective_rpc('profile_inspect_weights')
        if any(w['head_dtype'] != 'torch.bfloat16' or w['embedding_dtype'] != 'torch.bfloat16'
                or w['quantization'] is not None or w['model_revision'] != args.model_revision for w in workers):
            raise ValueError('B precision/revision mismatch')
        if wrapper.HF_model.config._commit_hash != args.model_revision or next(wrapper.HF_model.parameters()).dtype != torch.bfloat16:
            raise ValueError('A precision/revision mismatch')
        realignment = {}
        if manifest.get('migration'):
            from controller.server_training import bind_archived_realignment
            realignment = bind_archived_realignment(wrapper,
                manifest.get('realignment_profile', manifest['profile']),
                manifest['migration']['archived_realignment_sha256'])
        else:
            matrix, norm = wrapper._ensure_latent_realign_matrix(wrapper.HF_model, torch.device(args.device2), args)
            archived = load_file(str(Path(manifest['profile']) / 'shared/realignment.safetensors'))
            if not torch.equal(matrix.cpu(), archived['matrix']) or not torch.equal(norm.cpu(), archived['target_norm']):
                raise ValueError('Realignment differs from profile')
        write_json(root / f'preflight_{time.time_ns()}.json', dict(workers=workers,
            hf_dtype='torch.bfloat16', normalization=cfg["controller_normalization"], realignment_exact=True,
            normalization_fit_split='train', old_controller_weights_loaded=False, **realignment))
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        seen, episodes_seen = prior_coverage(root, last)
        checkpoints = manifest['validation_checkpoints']
        def evaluate(n):
            metrics = validation(wrapper, args, model, valid_items, root, n, report['summaries']['validation'],
                baseline_source=manifest.get('migration',{}).get('baseline_source','Verified profile validation grid'))
            tracker.log(dict(metrics, iteration=n))
            event(root, stage='validation_complete', iteration=n,
                accuracy=metrics['validation/PPO/accuracy'], mean_K=metrics['validation/PPO/mean_K'])
        for n in checkpoints:
            if n < last and completed(root / 'validation' / f'iteration{n:03d}') is None:
                raise ValueError('Missing earlier committed validation')
        if last in checkpoints:
            evaluate(last)
        ties = (d['q_targets'][:, :, 0] == d['q_targets'][:, :, 1] if cfg['tie_weight']
                else torch.zeros((pool_size, 7), dtype=torch.bool, device=d['features'].device))
        original_update = core.ppo_update
        try:
            for n in range(last + 1, cfg['iterations'] + 1):
                plan = sampler.plan(n)
                index = torch.tensor(plan['indices'], device=args.device)
                def update(model, opt, x, episode, teacher, masks, config):
                    def log(step, values):
                        tracker.log(dict(values, optimizer_step=(n-1)*16+step, iteration=n))
                    started = time.perf_counter()
                    result = logged_ppo_update(model, opt, x, episode, teacher, masks, config, log)
                    torch.cuda.synchronize()
                    update.seconds = time.perf_counter() - started
                    return result
                core.ppo_update = update
                event_count = len((root / 'events.jsonl').read_text().splitlines()) if (root / 'events.jsonl').exists() else 0
                path = core.run_iteration(wrapper, args, root, [items[i] for i in plan['indices']],
                    model, optimizer, d['features'][index], ties[index], cfg, plan, 0, auxiliary_pool_size=pool_size)
                result = json.loads((path / 'result.json').read_text())
                episode = load_file(str(path / 'episodes.safetensors'))
                outcomes = read_outcomes(path)
                seen.update(r['source_id'] for r in outcomes)
                episodes_seen += 32
                new_events = [json.loads(s) for s in (root / 'events.jsonl').read_text().splitlines()[event_count:]]
                metrics = dict(rollout_metrics(episode), **outcome_metrics(outcomes))
                metrics.update({'iteration/loss_total': result['loss'], 'iteration/committed': 1,
                    'train/accuracy': result['correct']/32, 'train/mean_K': result['mean_selected_K'],
                    'time/iteration_seconds': result['seconds_including_live_A_B'],
                    'time/B_generate_seconds': sum(e['seconds'] for e in new_events if e.get('stage') == 'B_cohort_complete'),
                    'time/PPO_update_seconds': update.seconds,
                    'budget/cohort_physical_steps_max': max(result['cohort_physical_steps']),
                    'data/episodes_seen': episodes_seen, 'data/unique_PPO_questions_seen': len(seen),
                    'data/unique_auxiliary_questions': pool_size, 'data/train_pool_size': pool_size,
                    'data/sampler_epoch': plan['epoch'], 'data/iteration_in_epoch': plan['iteration_in_epoch'],
                    'validation/available': 0})
                metrics.update({'train/postupdate_'+k: v for k, v in
                    encoder_diagnostics(model, episode['features'].to(args.device),
                        episode['visited'].to(args.device)).items()})
                tracker.log(dict(metrics, iteration=n))
                if n in checkpoints:
                    evaluate(n)
        finally:
            core.ppo_update = original_update
        validations = []
        for n in checkpoints:
            p = completed(root / 'validation' / f'iteration{n:03d}')
            v = json.loads((p / 'metrics.json').read_text())
            validations.append(dict(iteration=n, accuracy=v['validation/PPO/accuracy'], mean_K=v['validation/PPO/mean_K']))
        best = min(validations, key=lambda x: (-x['accuracy'], x['mean_K'], x['iteration']))
        write_json(root / 'training_finished.json', dict(status='complete', iterations=cfg['iterations'],
            warmstart_epochs=cfg['warmstart_epochs'], warmstart_skipped=cfg['warmstart_epochs'] == 0,
            train_episodes=cfg['iterations']*32, validation_B_completions=len(checkpoints)*128,
            validations=validations, validation_checkpoints=checkpoints, best_validation=best,
            checkpoint_selection='Validation accuracy, then meanK, then earlier iteration; no test selection',
            controller_normalization=cfg["controller_normalization"], old_controller_weights_loaded=False,
            test300_used=False, fixed_baseline_generations=0, model_weights_frozen=True))
        event(root, stage='training_complete', iterations=cfg['iterations'])
        code = 0
    finally:
        tracker.finish(code)
