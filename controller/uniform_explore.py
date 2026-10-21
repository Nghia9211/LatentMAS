"""Uniform stopping-budget initialization, without changing terminal rewards.

The checkpoint hazard is part of the policy forward, used by rollout AND PPO.
It uses only existing k/80. No forced actions, future states or gold labels.
"""
from pathlib import Path
import torch
from torch.distributions import Categorical, kl_divergence
from controller.normalized import NormalizedPPOController

SCHEMA = 'normalized_uniform_budget_init_tie_ramp_v1'


def uniform_log_prior(x):
    j_float = x[..., -1] * 8  # Existing metadata k/80, checkpoints k=10*j.
    j = j_float.round().long()
    if (not torch.isfinite(j_float).all() or not torch.allclose(j_float, j.float(), atol=1e-6, rtol=0)
            or ((j < 0) | (j > 7)).any()):
        raise ValueError('Expected k/80 for K10..70; zero allowed only for loss padding')
    # Dummy unvisited PPO slots have k=0. Their finite logits are masked in loss.
    handoff = torch.where(j == 0, .5, 1. / (9-j).float())
    return torch.stack((handoff.log(), torch.log1p(-handoff)), dim=-1)


def tie_weight(iteration, base=.1):
    if type(iteration) is not int or iteration < 1 or base < 0:
        raise ValueError('Invalid schedule')
    # Pilot8: quality/exploration first4, then .025/.05/.075/.1 at5..8.
    return base * min(1., max(0., (iteration-4)/4))


class UniformBudgetPPOController(NormalizedPPOController):
    def __init__(self, input_dim, width=128, *, eps=1e-5, projection_norm=True):
        super().__init__(input_dim, width, eps=eps, projection_norm=projection_norm)
        self.register_buffer('uniform_budget_version', torch.tensor(1, dtype=torch.long))
        self.register_buffer('exploration_iterations', torch.tensor(0, dtype=torch.long))
        self.reset_actor()

    @torch.no_grad()
    def reset_actor(self):
        self.actor.weight.zero_()
        self.actor.bias.zero_()

    def forward(self, x):
        residual, value = super().forward(x)
        return residual + uniform_log_prior(x), value

    def load_warmstart(self, warm):
        super().load_warmstart(warm)
        # Preserve trained encoder/stats; discard STOP-biased actor, fresh critic.
        self.reset_actor()
        self.exploration_iterations.zero_()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if ('uniform_budget_version' not in state_dict or 'exploration_iterations' not in state_dict
                or int(state_dict['uniform_budget_version']) != 1
                or int(state_dict['exploration_iterations']) < 0):
            raise ValueError('Require a versioned uniform-budget checkpoint, not old normalized/raw weights')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def load_uniform_checkpoint(path, device='cpu'):
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    state = payload['model']
    width, dim = state['encoder.0.weight'].shape
    model = UniformBudgetPPOController(dim, width, eps=float(state['normalizer.epsilon']),
                                     projection_norm=bool(state['projection_normalized']))
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def budget_probabilities(model, features):
    """Train-profile diagnostic only, not an online observation or future reward."""
    if features.ndim != 3 or features.shape[1] != 7:
        raise ValueError('Need full train observation grid')
    p = model(features)[0].softmax(-1)
    reach = torch.ones(len(features), device=features.device)
    stopped = []
    for j in range(7):
        stopped.append(reach * p[:, j, 0])
        reach = reach * p[:, j, 1]
    return torch.stack(stopped+[reach], dim=-1)


@torch.no_grad()
def exact_visited_kl(model, x, old_logits, visited):
    if not visited.any():
        raise ValueError('No visited states')
    new_logits = model(x)[0]
    return float(kl_divergence(Categorical(logits=old_logits[visited]),
                               Categorical(logits=new_logits[visited])).mean())
