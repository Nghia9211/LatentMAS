"""Versioned normalization for NEW controllers; historical raw models stay intact.

Input is still raw [h_k, h0, delta_checkpoint, phi4, k/80]. Hidden blocks use
per-observation, non-affine LayerNorm; phi statistics are fit on train only and
checkpointed. No batch statistics, labels, future states, or new features.
"""
import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from controller.ppo import PPOController
from controller.warmstart import WarmStartController

SCHEMA = 'block_ln_phi_train_zscore_v1'


class ObservationNormalizer(nn.Module):
    def __init__(self, input_dim, eps=1e-5):
        super().__init__()
        if input_dim < 8 or (input_dim - 5) % 3 or not 0 < eps < 1:
            raise ValueError('Expected 3*hidden_dim+5 features and positive small epsilon')
        self.input_dim = input_dim
        self.hidden_dim = (input_dim - 5) // 3
        self.eps = float(eps)
        self.register_buffer('phi_mean', torch.zeros(4))
        self.register_buffer('phi_scale', torch.ones(4))
        self.register_buffer('fitted', torch.tensor(False))
        self.register_buffer('schema_version', torch.tensor(1, dtype=torch.int64))
        self.register_buffer('epsilon', torch.tensor(eps, dtype=torch.float64))

    @torch.no_grad()
    def fit(self, features, *, source_split):
        if source_split != 'train':
            raise ValueError('Normalization statistics may only be fit on train')
        if bool(self.fitted):
            raise ValueError('Statistics are frozen after fitting; create a new normalizer')
        self._check_input(features)
        phi = features.detach().float()[..., -5:-1].reshape(-1, 4)
        if not len(phi):
            raise ValueError('Empty training features')
        mean = phi.mean(0)
        std = phi.std(0, unbiased=False)
        # A constant train feature must not amplify inference deviations by 1/eps.
        scale = torch.where(std >= self.eps, std, torch.ones_like(std))
        self.phi_mean.copy_(mean.to(self.phi_mean))
        self.phi_scale.copy_(scale.to(self.phi_scale))
        self.fitted.fill_(True)
        return self

    def _check_input(self, x):
        if x.ndim < 2 or x.shape[-1] != self.input_dim or not torch.isfinite(x).all():
            raise ValueError('Expected finite [..., observation_dim] input')

    def forward(self, x):
        self._check_input(x)
        if not bool(self.fitted):
            raise ValueError('Fit train-only statistics or load a normalized checkpoint first')
        if (not torch.isfinite(self.phi_mean).all() or not torch.isfinite(self.phi_scale).all()
                or not (self.phi_scale > 0).all()):
            raise ValueError('Invalid checkpointed normalization statistics')
        raw = x.float()
        d = self.hidden_dim
        blocks = [F.layer_norm(raw[..., i*d:(i+1)*d], (d,), eps=self.eps)
                  for i in range(3)]
        phi = (raw[..., -5:-1] - self.phi_mean) / self.phi_scale
        # No normalization or change to k/80; padding remains loss bookkeeping.
        return torch.cat([*blocks, phi, raw[..., -1:]], dim=-1)


def _add_projection_norm(encoder, width, eps, enabled):
    if not enabled:
        return encoder
    # Preserve initial Linear weights/RNG and encoder[0].in_features used online.
    return nn.Sequential(encoder[0],
        nn.LayerNorm(width, eps=eps, elementwise_affine=False), encoder[1])


class _NormalizedController:
    """Shared strict checkpoint guard: never silently reinterpret old raw weights."""
    def _setup_normalization(self, input_dim, width, eps, projection_norm):
        self.normalizer = ObservationNormalizer(input_dim, eps)
        self.register_buffer('projection_normalized', torch.tensor(bool(projection_norm)))
        self.encoder = _add_projection_norm(self.encoder, width, eps, projection_norm)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        required = ('normalizer.phi_mean', 'normalizer.phi_scale', 'normalizer.fitted',
                    'normalizer.schema_version', 'normalizer.epsilon', 'projection_normalized')
        if any(k not in state_dict for k in required):
            raise ValueError('Raw checkpoint cannot be loaded as a normalized controller')
        if (int(state_dict['normalizer.schema_version']) != 1
                or not bool(state_dict['normalizer.fitted'])
                or float(state_dict['normalizer.epsilon']) != self.normalizer.eps
                or bool(state_dict['projection_normalized']) != bool(self.projection_normalized)):
            raise ValueError('Normalization architecture/statistics mismatch')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class NormalizedWarmStartController(_NormalizedController, WarmStartController):
    def __init__(self, input_dim, width=128, *, eps=1e-5, projection_norm=True):
        super().__init__('raw_mlp', input_dim, width)
        self._setup_normalization(input_dim, width, eps, projection_norm)

    def forward(self, x):
        return super().forward(self.normalizer(x))


class NormalizedPPOController(_NormalizedController, PPOController):
    def __init__(self, input_dim, width=128, *, eps=1e-5, projection_norm=True):
        super().__init__(input_dim, width)
        self._setup_normalization(input_dim, width, eps, projection_norm)

    def forward(self, x):
        return super().forward(self.normalizer(x))

    def load_warmstart(self, warm):
        if (not isinstance(warm, NormalizedWarmStartController)
                or bool(warm.projection_normalized) != bool(self.projection_normalized)
                or warm.normalizer.eps != self.normalizer.eps or not bool(warm.normalizer.fitted)):
            raise ValueError('Require matching normalized warm-start')
        super().load_warmstart(warm)
        self.normalizer.load_state_dict(copy.deepcopy(warm.normalizer.state_dict()))


def load_normalized_checkpoint(path, device='cpu'):
    """Inference loader; uses saved train statistics, never refits on test inputs."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    state = payload['model']
    required = ('normalizer.epsilon', 'projection_normalized', 'encoder.0.weight')
    if any(k not in state for k in required):
        raise ValueError('Not a normalized PPO checkpoint')
    width, input_dim = state['encoder.0.weight'].shape
    model = NormalizedPPOController(input_dim, width,
        eps=float(state['normalizer.epsilon']), projection_norm=bool(state['projection_normalized']))
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def encoder_diagnostics(model, features, visited=None):
    x = features if visited is None else features[visited]
    if not x.numel():
        return {}
    normalized = model.normalizer(x)
    pre = model.encoder[:-1](normalized)
    z = model.encoder(normalized)
    return dict(encoder_near_zero_fraction=float((z.abs() < 1e-6).float().mean()),
        encoder_all_zero_fraction=float((z.abs().amax(-1) < 1e-6).float().mean()),
        pre_gelu_negative_fraction=float((pre < 0).float().mean()),
        pre_gelu_rms=float(pre.square().mean().sqrt()))
