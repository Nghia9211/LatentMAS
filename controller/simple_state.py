"""Versioned causal h_k + relative checkpoint-L2 + k/80 for LIVE PPO."""
import copy
import torch
from controller.ppo import PPOController
from controller.warmstart import WarmStartController
from controller.three_group import ThreeGroupObservationNormalizer, ThreeGroupControllerMixin, ThreeGroupCohortDecisions, patched_decision_adapter, load_three_group_checkpoint
SCHEMA = 'hk_l2_k_train_zscore_v1'
DIM = 2562
HIDDEN = 2560

def select_profile_features(x):
    if x.shape[-1] != 7685 or not torch.isfinite(x).all():
        raise ValueError('Expected verified full raw observation')
    return torch.cat([x[..., :HIDDEN], x[..., -4:-3], x[..., -1:]], -1).contiguous()

def raw_observation(hidden, k):
    if k not in range(10, 81, 10) or hidden.ndim != 2 or hidden.shape[0] < k + 1 or (hidden.shape[1] != HIDDEN):
        raise ValueError('Need causal h0..hk with hidden2560 and checkpoint10..80')
    h = hidden[k].detach().float()
    prev = hidden[k - 10].detach().float()
    if not torch.isfinite(h).all() or not torch.isfinite(prev).all():
        raise ValueError('Nonfinite hidden')
    l2 = torch.linalg.vector_norm(h - prev) / (torch.linalg.vector_norm(prev) + 1e-08)
    return torch.cat([h, l2[None], h.new_tensor([k / 80])])

class SimpleObservationNormalizer(ThreeGroupObservationNormalizer):
    scalar_name = 'l2'
    schema_version_value = 2

class _Simple(ThreeGroupControllerMixin):
    normalizer_type = SimpleObservationNormalizer

class SimpleWarmStartController(_Simple, WarmStartController):

    def __init__(self, input_dim=DIM, width=128, *, eps=1e-05, projection_norm=True):
        super().__init__('raw_mlp', input_dim, width)
        self.setup(input_dim, width, eps, projection_norm)

class SimplePPOController(_Simple, PPOController):

    def __init__(self, input_dim=DIM, width=128, *, eps=1e-05, projection_norm=True):
        super().__init__(input_dim, width)
        self.setup(input_dim, width, eps, projection_norm)

    def load_warmstart(self, warm):
        if not isinstance(warm, SimpleWarmStartController) or not bool(warm.normalizer.fitted) or warm.normalizer.eps != self.normalizer.eps or (bool(warm.projection_normalized) != bool(self.projection_normalized)):
            raise ValueError('Require matching simple-state random prototype')
        super().load_warmstart(warm)
        self.normalizer.load_state_dict(copy.deepcopy(warm.normalizer.state_dict()))

class SimpleCohortDecisions(ThreeGroupCohortDecisions):
    observation_builder = staticmethod(raw_observation)

def live_adapters():
    return patched_decision_adapter(SimpleCohortDecisions)

def load_simple_checkpoint(path, device='cpu'):
    return load_three_group_checkpoint(path, SimplePPOController, device)
