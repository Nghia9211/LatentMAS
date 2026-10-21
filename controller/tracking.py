"""Opt-in W&B scalar logging; no prompts, tensors, checkpoints or code uploads."""
import importlib
import json
import math
from pathlib import Path
import uuid
import warnings

from controller.profile import file_hash, write_json

PUBLIC_CONFIG_KEYS = {
    'model_name', 'model_revision', 'task', 'method', 'num_agents', 'prompt', 'think',
    'latent_space_realign', 'temperature', 'top_p', 'max_new_tokens', 'seed',
    'generate_bs', 'latent_steps', 'vllm_dtype', 'width', 'warmstart_epochs',
    'iterations', 'optimization_epochs', 'minibatch_questions', 'learning_rate',
    'clip', 'value_weight', 'entropy_weight', 'tie_weight', 'grad_clip', 'gamma',
    'time_penalty', 'runner', 'validation_samples', 'validation_seed',
    'train_samples', 'sampler_seed', 'iterations_per_epoch',
}


def require_wandb(mode):
    if mode not in ('disabled', 'offline', 'online'):
        raise ValueError('W&B mode must be disabled, offline or online')
    if mode == 'disabled':
        return None
    try:
        return importlib.import_module('wandb')
    except ImportError as error:
        raise RuntimeError('Install optional tracking dependency: python -m pip install -r requirements-tracking.txt') from error


class TrainingTracker:
    def __init__(self, root, *, mode='disabled', project='latentmas-controller',
                 entity=None, name=None, config=None, sdk=None, run_id=None):
        if mode not in ('disabled', 'offline', 'online'):
            raise ValueError('W&B mode must be disabled, offline or online')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.remote_failed = False
        self.run = None
        self.attempt = 1
        self.log_path = self.root / 'training_metrics.jsonl'
        if self.log_path.exists():
            rows = [json.loads(s) for s in self.log_path.read_text().splitlines() if s.strip()]
            self.attempt = max((r.get('tracking/attempt', 0) for r in rows), default=0) + 1
        metadata_path = self.root / 'wandb_tracking.json'
        identity = dict(mode=mode, project=project, entity=entity)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if any(metadata[k] != v for k, v in identity.items()):
                raise ValueError('Tracking identity changed; use a separate output run')
            if run_id is not None and run_id != metadata['run_id']:
                raise ValueError('Explicit W&B run ID differs from saved identity')
            run_id = metadata['run_id']
        else:
            run_id = run_id or uuid.uuid4().hex[:12]
            write_json(metadata_path, dict(**identity, run_id=run_id,
                scalar_only=True, code_upload=False, console_upload=False,
                logger_source_sha256=file_hash(Path(__file__))))
        sdk = require_wandb(mode) if sdk is None else sdk
        if mode != 'disabled':
            public_config = {k: v for k, v in (config or {}).items()
                             if k in PUBLIC_CONFIG_KEYS and isinstance(v, (str, int, float, bool, type(None)))}
            self.run = sdk.init(project=project, entity=entity, name=name or self.root.name,
                id=run_id, mode=mode, dir=str(self.root),
                resume='allow' if mode == 'online' else None,
                config=public_config, save_code=False,
                settings=sdk.Settings(console='off', disable_git=True, disable_code=True, init_timeout=30))
            self.run.define_metric('optimizer_step')
            self.run.define_metric('iteration')
            for pattern in ('loss/*', 'ppo/*'):
                self.run.define_metric(pattern, step_metric='optimizer_step')
            for pattern in ('iteration/*', 'train/*', 'time/*', 'budget/*', 'handoff/*',
                            'rollout/*', 'output/*', 'data/*', 'validation/*'):
                self.run.define_metric(pattern, step_metric='iteration')

    def log(self, fields):
        row = {'tracking/attempt': self.attempt}
        for key, value in fields.items():
            if not isinstance(value, (int, float, bool)):
                raise TypeError('Only scalar numeric training metrics may be logged')
            if math.isfinite(value):
                row[key] = value
        with self.log_path.open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        if self.run is not None and not self.remote_failed:
            try:
                # Custom x axes avoid mixing iteration and optimizer-step counters.
                self.run.log(row)
            except Exception as error:
                self.remote_failed = True
                warnings.warn(f'W&B logging stopped ({type(error).__name__}); scalar logs remain local.', RuntimeWarning)

    def finish(self, exit_code=0):
        if self.run is not None:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception as error:
                warnings.warn(f'W&B finish failed ({type(error).__name__}); local logs retained.', RuntimeWarning)
