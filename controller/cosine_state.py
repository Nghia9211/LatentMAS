"""Versioned causal h_k + checkpoint cosine distance + k/80 for LIVE PPO."""
import copy
import torch
from torch.nn import functional as F
from controller.ppo import PPOController
from controller.warmstart import WarmStartController
from controller.three_group import ThreeGroupObservationNormalizer, ThreeGroupControllerMixin, ThreeGroupCohortDecisions, patched_decision_adapter, load_three_group_checkpoint
SCHEMA = 'hk_cosine10_k_train_zscore_v1'
DIM = 2562
HIDDEN = 2560

def select_profile_features(x):
    """Exact checkpoint10 cosine from ordered full causal profile states."""
    if x.ndim < 3 or x.shape[-2:] != (7, 7685) or (not torch.isfinite(x).all()):
        raise ValueError('Expected ordered seven verified full raw observations')
    h = x[..., :HIDDEN]
    previous = torch.cat([x[..., 0:1, HIDDEN:2 * HIDDEN], h[..., :-1, :]], dim=-2)
    if not torch.allclose(h - previous, x[..., 2 * HIDDEN:3 * HIDDEN], rtol=1e-05, atol=1e-05):
        raise ValueError('Profile checkpoint history is inconsistent')
    distance = (1 - F.cosine_similarity(h, previous, dim=-1, eps=1e-08)).clamp(0, 2)[..., None]
    return torch.cat([h, distance, x[..., -1:]], -1).contiguous()

def raw_observation(hidden, k):
    if k not in range(10, 81, 10) or hidden.ndim != 2 or hidden.shape[0] < k + 1 or (hidden.shape[1] != HIDDEN):
        raise ValueError('Need causal h0..hk with hidden2560 and checkpoint10..80')
    h = hidden[k].detach().float()
    prev = hidden[k - 10].detach().float()
    if not torch.isfinite(h).all() or not torch.isfinite(prev).all():
        raise ValueError('Nonfinite hidden')
    distance = (1 - F.cosine_similarity(h, prev, dim=0, eps=1e-08)).clamp(0, 2)
    return torch.cat([h, distance[None], h.new_tensor([k / 80])])

class CosineObservationNormalizer(ThreeGroupObservationNormalizer):
    scalar_name = 'cosine'
    schema_version_value = 3

class _Cosine(ThreeGroupControllerMixin):
    normalizer_type = CosineObservationNormalizer

class CosineWarmStartController(_Cosine, WarmStartController):

    def __init__(self, input_dim=DIM, width=128, *, eps=1e-05, projection_norm=True):
        super().__init__('raw_mlp', input_dim, width)
        self.setup(input_dim, width, eps, projection_norm)

class CosinePPOController(_Cosine, PPOController):

    def __init__(self, input_dim=DIM, width=128, *, eps=1e-05, projection_norm=True):
        super().__init__(input_dim, width)
        self.setup(input_dim, width, eps, projection_norm)

    def load_warmstart(self, warm):
        if not isinstance(warm, CosineWarmStartController) or not bool(warm.normalizer.fitted) or warm.normalizer.eps != self.normalizer.eps or (bool(warm.projection_normalized) != bool(self.projection_normalized)):
            raise ValueError('Require matching cosine-state random prototype')
        super().load_warmstart(warm)
        self.normalizer.load_state_dict(copy.deepcopy(warm.normalizer.state_dict()))

class CosineCohortDecisions(ThreeGroupCohortDecisions):
    observation_builder = staticmethod(raw_observation)

def live_adapters():
    return patched_decision_adapter(CosineCohortDecisions)

def load_cosine_checkpoint(path, device='cpu'):
    return load_three_group_checkpoint(path, CosinePPOController, device)
