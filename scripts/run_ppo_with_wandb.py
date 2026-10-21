#!/usr/bin/env python3
"""W&B entrypoint for matched PPO or rotating pooled PPO.

Keeps historical runner files and source locks unchanged. Only replaces the
optimizer-update function with an arithmetic-equivalent instrumented function.
All normal runner flags are forwarded. Disabled mode needs no W&B installation.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from controller.tracking import TrainingTracker, require_wandb
from controller.training import logged_ppo_update
from controller.metrics import rollout_metrics, outcome_metrics, prior_coverage


class PPOLoggingSession:
    def __init__(self, root, options, runner):
        self.root, self.options, self.runner = Path(root), options, runner
        self.tracker = None
        self.previous = None
        self.current = None
        self.last_loss = None
        self.closed = False
        self.rollout_summary = {}
        self.timings = {}
        self.rows = []
        self.physical_steps = []
        self.seen_ids = set()
        self.episodes_seen = 0
        self.coverage_known = True

    def measured(self, name, function, *args, **kwargs):
        if name == 'reset_cache':
            self.timings, self.rows, self.physical_steps = {}, [], []
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = function(*args, **kwargs)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        labels = {'render_batch': 'render_seconds', 'online_rollout': 'A_controller_seconds',
                  'prefixes_for_b': 'communication_prefix_seconds', 'reset_cache': 'reset_cache_seconds'}
        if name in labels:
            key = 'time/' + labels[name]
            self.timings[key] = self.timings.get(key, 0) + seconds
        if name == 'online_rollout':
            self.physical_steps.append(result[2])
        if name == 'decode_cohort':
            purpose = kwargs.get('purpose', args[7] if len(args) > 7 else None)
            if purpose == 'live_train_reward':
                rows, generated_seconds = result
                self.rows.extend(rows)
                for key, value in [('time/B_generate_seconds', generated_seconds),
                                   ('time/B_outcome_archive_seconds', max(0, seconds - generated_seconds))]:
                    self.timings[key] = self.timings.get(key, 0) + value
        return result

    def update(self, model, optimizer, x, episode, teacher_x, ties, cfg):
        if self.tracker is None:
            manifest = json.loads((self.root / 'manifest.json').read_text())
            if self.runner == 'matched':
                config = manifest['config']
                completed = []
                for marker in sorted((self.root / 'train').glob('iteration*/complete.json')):
                    completed.append(int(marker.parent.name.replace('iteration', '')))
                self.previous = max(completed, default=0)
                self.seen_ids, self.episodes_seen = prior_coverage(self.root, self.previous)
            else:
                config = manifest['parent_config']
                self.previous = manifest['parent_iteration']
                if manifest.get('training'):
                    self.seen_ids, self.episodes_seen = prior_coverage(Path(manifest['training']), self.previous)
                else:
                    self.coverage_known = False
                if self.runner == 'pooled':
                    local = 0
                    while (self.root / 'train' / f'iteration{local + 1:03d}' / 'complete.json').exists():
                        local += 1
                    local_ids, local_episodes = prior_coverage(self.root, local)
                    self.seen_ids.update(local_ids)
                    self.episodes_seen += local_episodes
                    self.previous += local
                    config = dict(config, iterations=manifest['iterations'],
                        train_samples=manifest['train_manifest']['count'],
                        sampler_seed=manifest['sampler']['seed'],
                        iterations_per_epoch=manifest['sampler']['iterations_per_epoch'])
            self.auxiliary_questions = len(set(manifest.get('parent_train_ids',
                manifest.get('train_manifest', {}).get('source_ids', []))))
            self.tracker = TrainingTracker(self.root,
                mode=self.options.wandb_mode, project=self.options.wandb_project,
                entity=self.options.wandb_entity, name=self.options.wandb_name,
                config=dict(config, runner=self.runner))
        self.current = self.previous + 1
        self.rollout_summary = rollout_metrics(episode)
        steps_per_iteration = cfg['optimization_epochs'] * (32 // cfg['minibatch_questions'])

        def record(local_step, metrics):
            self.tracker.log(dict(metrics, optimizer_step=self.previous * steps_per_iteration + local_step,
                                  iteration=self.current))

        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        started = time.perf_counter()
        loss, updates = logged_ppo_update(model, optimizer, x, episode, teacher_x, ties, cfg, record)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        self.timings['time/PPO_update_seconds'] = time.perf_counter() - started
        self.last_loss = loss
        self.previous = self.current
        return loss, updates

    def event(self, original, root, **fields):
        original(root, **fields)
        if self.tracker is None or fields.get('stage') not in ('PPO_iteration_complete', 'one_iteration_complete'):
            return
        iteration = fields.get('iteration', fields.get('total_controller_iterations'))
        extra = dict(self.rollout_summary, **self.timings, **outcome_metrics(self.rows))
        if self.rows:
            self.seen_ids.update(r['source_id'] for r in self.rows)
            self.episodes_seen += len(self.rows)
            extra['data/episodes_seen'] = self.episodes_seen
            if self.coverage_known:
                extra['data/unique_PPO_questions_seen'] = len(self.seen_ids)
        extra.update({'data/coverage_complete': int(self.coverage_known),
                      'data/unique_auxiliary_questions': self.auxiliary_questions,
                      'validation/available': 0})
        if self.runner == 'pooled':
            extra.update({'data/sampler_epoch': fields['sampler_epoch'],
                'data/iteration_in_epoch': fields['iteration_in_epoch'],
                'data/train_pool_size': fields['pool_size']})
        if self.physical_steps:
            extra['budget/cohort_physical_steps_max'] = max(self.physical_steps)
        self.tracker.log(dict(extra, iteration=iteration,
            **{'iteration/loss_total': self.last_loss, 'iteration/committed': 1,
               'train/accuracy': fields['correct'] / 32,
               'train/mean_K': fields.get('mean_K', fields.get('mean_selected_K')),
               'time/iteration_seconds': fields.get('seconds', fields.get('seconds_including_live_A_B'))}))

    def close(self, exit_code):
        if not self.closed and self.tracker is not None:
            self.tracker.finish(exit_code)
        self.closed = True


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--runner', choices=('matched', 'pooled'), default='matched')
    parser.add_argument('--wandb-mode', choices=('disabled', 'offline', 'online'), default='disabled')
    parser.add_argument('--wandb-project', default='latentmas-controller')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--wandb-name')
    options, forwarded = parser.parse_known_args()
    if '--help' in forwarded or '-h' in forwarded:
        parser.print_help()
        print('\nOther arguments are forwarded to the selected legacy runner.\n')
    else:
        require_wandb(options.wandb_mode)  # Fail before LLM work if optional SDK is absent.
    destination = argparse.ArgumentParser(add_help=False)
    destination.add_argument('--out', type=Path)
    output, _ = destination.parse_known_args(forwarded)
    if output.out is None and '--help' not in forwarded and '-h' not in forwarded:
        parser.error('--out is required by the selected runner')
    if options.runner == 'matched':
        from scripts import run_matched_controller_ppo as driver
    else:
        from scripts import run_pooled_controller_ppo as driver
    # New runs snapshot the logging implementation as well as the legacy math.
    # Old manifests are not silently upgraded or overwritten on --resume.
    previous_sources = driver.SOURCES
    driver.SOURCES = list(driver.SOURCES) + [
        'controller/tracking.py', 'controller/metrics.py',
        'scripts/run_ppo_with_wandb.py']
    session = PPOLoggingSession(output.out or '.', options, options.runner)
    original_update, original_event = driver.ppo_update, driver.event
    driver.ppo_update = session.update
    driver.event = lambda root, **fields: session.event(original_event, root, **fields)
    measured = {}
    for name in ('reset_cache', 'render_batch', 'online_rollout', 'prefixes_for_b', 'decode_cohort'):
        function = getattr(driver, name)
        measured[name] = function
        setattr(driver, name, lambda *args, _name=name, _fn=function, **kwargs:
                session.measured(_name, _fn, *args, **kwargs))
    previous_argv = sys.argv
    sys.argv = [str(Path(driver.__file__))] + forwarded
    exit_code = 1
    try:
        driver.main()
        exit_code = 0
    finally:
        session.close(exit_code)
        sys.argv = previous_argv
        driver.ppo_update, driver.event = original_update, original_event
        for name, function in measured.items():
            setattr(driver, name, function)
        if previous_sources is not None:
            driver.SOURCES = previous_sources


if __name__ == '__main__':
    main()
