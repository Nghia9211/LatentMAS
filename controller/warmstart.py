"""Small frozen-LLM controller warm-start primitives; no LLM imports/calls.

Architecture convention follows design section4.7: common raw v, no LayerNorm,
C2 GELU bottleneck128; C3 same encoder plus unidirectional GRU128.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file

from controller.profile import GRID, PHI_SCHEMA, raw_observation, tensor_hash

CANDIDATES = ('raw_linear', 'raw_mlp', 'projection_gru')
ACTION_NAMES = ('HANDOFF', 'CONTINUE_10')


def supervision(correctness):
    """H=q(k), C=max observed q(k+10..80); labels only, never features."""
    values = torch.as_tensor(correctness, dtype=torch.float32)
    if values.ndim != 1 or len(values) != 8 or not torch.all((values == 0) | (values == 1)):
        raise ValueError('Need exactly8 binary full-grid outcomes')
    best = values.flip(0).cummax(0).values.flip(0)
    q = torch.stack((values[:-1], best[1:]), dim=-1)
    actions = (q[:, 0] < q[:, 1]).long()  # Ties -> HANDOFF0.
    return actions, q


def build_dataset(root):
    """Use ONLY last-layer h<=k + phi4 + k/80; retain each question as a unit.

    Caller must run full independent verification before this reader.
    """
    root = Path(root)
    split = json.loads((root/'split_manifest.json').read_text())
    schema = json.loads((root/'feature_schema.json').read_text())
    if schema['phi_schema'] != PHI_SCHEMA or schema['policy_metadata'] != ['k/80'] or schema['availability_masks']:
        raise ValueError('Unexpected observation schema')
    features, actions, q_targets, label_histogram = [], [], [], torch.zeros(2, dtype=torch.long)
    hidden_dim = None
    for source_id in split['source_ids']:
        base = root/'trajectories'/source_id
        marker = json.loads((base/'complete.json').read_text())
        dest = base/marker['attempt']
        states = load_file(str(dest/'states.safetensors'))
        hidden = states['hidden_last_position_all_layers'][:, -1].clone()
        if hidden.shape[0] != 81 or hidden.dtype != torch.bfloat16 or not torch.isfinite(hidden).all():
            raise ValueError('Invalid BF16 h0..80')
        hidden_dim = hidden.shape[-1] if hidden_dim is None else hidden_dim
        if hidden.shape[-1] != hidden_dim:
            raise ValueError('Mixed hidden dimensions')
        rows = [json.loads(line) for line in (dest/'b_outcomes.jsonl').read_text().splitlines()]
        if [row['k'] for row in rows] != list(GRID) or any(row['source_id'] != source_id or row['source_split'] != 'train' for row in rows):
            raise ValueError('Need ordered train-only complete grid')
        checkpoints = [json.loads(line) for line in (dest/'checkpoints.jsonl').read_text().splitlines()]
        x = torch.stack([raw_observation(hidden, k) for k in GRID[:-1]])
        for i, k in enumerate(GRID[:-1]):
            if checkpoints[i]['k'] != k:
                raise ValueError('Checkpoint order mismatch')
            torch.testing.assert_close(x[i, -5:-1], torch.tensor(checkpoints[i]['phi']), atol=1e-6, rtol=1e-5)
        a, q = supervision([int(row['strict_correct']) for row in rows])
        archived = json.loads((dest/'train_targets.json').read_text())['checkpoint_actions']
        if [ACTION_NAMES[int(value)] for value in a] != [archived[str(k)] for k in GRID[:-1]]:
            raise ValueError('Target mismatch')
        features.append(x)
        actions.append(a)
        q_targets.append(q)
        label_histogram += torch.bincount(a, minlength=2)
    x, a, q = torch.stack(features), torch.stack(actions), torch.stack(q_targets)
    if x.shape != (32, 7, 3*hidden_dim+5) or not torch.isfinite(x).all() or x.requires_grad:
        raise ValueError('Unexpected pilot32 causal dataset')
    metadata = dict(source_ids=split['source_ids'], questions=len(x), decision_checkpoints=list(GRID[:-1]),
        forced_cap=80, hidden_dim=hidden_dim, observation_dim=x.shape[-1],
        observation='[h_k,h0,h_k-h_(k-10),phi4,k/80]', normalization='none',
        phi_schema=PHI_SCHEMA, metadata=['k/80'], availability_masks=False,
        features_sha256=tensor_hash(x), actions_sha256=tensor_hash(a), q_targets_sha256=tensor_hash(q),
        label_counts=dict(zip(ACTION_NAMES, label_histogram.tolist())),
        question_batching=True, policy_inputs_exclude_gold_b_outcomes_future_timing=True)
    return x, a, q, metadata


class WarmStartController(nn.Module):
    """Actor2 and Q2 heads; Q is NOT the PPO current-policy value critic."""
    def __init__(self, candidate, input_dim, width=128):
        super().__init__()
        if candidate not in CANDIDATES or input_dim < 1 or width < 1:
            raise ValueError('Invalid architecture')
        self.candidate = candidate
        self.encoder = nn.Identity() if candidate == 'raw_linear' else nn.Sequential(nn.Linear(input_dim, width), nn.GELU())
        output_dim = input_dim if candidate == 'raw_linear' else width
        self.memory = nn.GRU(width, width, batch_first=True, bidirectional=False) if candidate == 'projection_gru' else None
        self.actor = nn.Linear(output_dim, 2)
        self.q_heads = nn.Linear(output_dim, 2)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] < 1:
            raise ValueError('Expected [questions,actual checkpoints,features]')
        z = self.encoder(x)
        if self.memory is not None:
            z, _ = self.memory(z)  # Fresh zero state per question, every call.
        return self.actor(z), self.q_heads(z)


def warmstart_loss(actor_logits, q_values, action_targets, q_targets, q_weight=1.0):
    if q_weight < 0 or actor_logits.shape != q_values.shape or q_values.shape != q_targets.shape or action_targets.shape != q_values.shape[:-1]:
        raise ValueError('Loss shape/coefficient mismatch')
    actor = F.cross_entropy(actor_logits.reshape(-1, 2), action_targets.reshape(-1))
    q_loss = F.smooth_l1_loss(q_values, q_targets, beta=1.0)
    return actor + q_weight*q_loss, dict(actor_ce=actor, q_huber=q_loss)


def check_gradients(model):
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    if any(g is None for g in gradients) or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('Missing/non-finite controller gradient')
    return any(bool(torch.count_nonzero(g)) for g in gradients)


@torch.no_grad()
def loss_diagnostics(model, x, actions, targets, q_weight):
    actor, q = model(x)
    total, pieces = warmstart_loss(actor, q, actions, targets, q_weight)
    if not all(torch.isfinite(value) for value in (total, *pieces.values())):
        raise FloatingPointError('Non-finite evaluation loss')
    return dict(loss=float(total), actor_ce=float(pieces['actor_ce']), q_huber=float(pieces['q_huber']),
                training_action_accuracy=float((actor.argmax(-1) == actions).float().mean()))
