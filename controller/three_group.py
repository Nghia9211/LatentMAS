"""Shared runtime for versioned [h_k, scalar, k/80] controllers.

Observation construction stays in each schema module. Buffer names and layer
ordering are preserved so existing L2 and cosine checkpoints remain loadable.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from controller.matched import CohortDecisions

HIDDEN = 2560
DIM = HIDDEN + 2


class ThreeGroupObservationNormalizer(nn.Module):
    scalar_name = None
    schema_version_value = None

    def __init__(self, input_dim=DIM, eps=1e-5):
        super().__init__()
        if input_dim != DIM or not 0 < eps < 1:
            raise ValueError("Expected three-group state dimension 2562")
        if self.scalar_name is None or self.schema_version_value is None:
            raise TypeError("Use a versioned observation normalizer")
        self.input_dim = input_dim
        self.hidden_dim = HIDDEN
        self.eps = float(eps)
        self.register_buffer(self.scalar_name + "_mean", torch.zeros(1))
        self.register_buffer(self.scalar_name + "_scale", torch.ones(1))
        self.register_buffer("fitted", torch.tensor(False))
        self.register_buffer(
            "schema_version",
            torch.tensor(self.schema_version_value, dtype=torch.int64),
        )
        self.register_buffer("epsilon", torch.tensor(eps, dtype=torch.float64))

    def check(self, x):
        if x.ndim < 2 or x.shape[-1] != DIM or not torch.isfinite(x).all():
            raise ValueError("Expected finite [..., 2562] observations")

    @torch.no_grad()
    def fit(self, x, *, source_split):
        if source_split != "train" or bool(self.fitted):
            raise ValueError("Fit once on train only")
        self.check(x)
        values = x.detach().float()[..., -2:-1].reshape(-1, 1)
        if not len(values):
            raise ValueError("Empty training observations")
        std = values.std(0, unbiased=False)
        getattr(self, self.scalar_name + "_mean").copy_(values.mean(0))
        getattr(self, self.scalar_name + "_scale").copy_(
            torch.where(std >= self.eps, std, torch.ones_like(std))
        )
        self.fitted.fill_(True)
        return self

    def forward(self, x):
        self.check(x)
        mean = getattr(self, self.scalar_name + "_mean")
        scale = getattr(self, self.scalar_name + "_scale")
        if (
            not bool(self.fitted)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError("Require valid frozen train-only statistics")
        x = x.float()
        return torch.cat(
            [
                F.layer_norm(x[..., :HIDDEN], (HIDDEN,), eps=self.eps),
                (x[..., -2:-1] - mean) / scale,
                x[..., -1:],
            ],
            -1,
        )


class ThreeGroupControllerMixin:
    normalizer_type = None

    def setup(self, input_dim, width, eps, projection_norm):
        self.normalizer = self.normalizer_type(input_dim, eps)
        self.register_buffer(
            "projection_normalized", torch.tensor(bool(projection_norm))
        )
        if projection_norm:
            self.encoder = nn.Sequential(
                self.encoder[0],
                nn.LayerNorm(width, eps=eps, elementwise_affine=False),
                self.encoder[1],
            )

    def forward(self, x):
        return super().forward(self.normalizer(x))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        prefix = "normalizer." + self.normalizer.scalar_name
        required = [
            prefix + "_mean",
            prefix + "_scale",
            "normalizer.fitted",
            "normalizer.schema_version",
            "normalizer.epsilon",
            "projection_normalized",
        ]
        if any(key not in state_dict for key in required):
            raise ValueError("Not a matching three-group normalized checkpoint")
        if (
            int(state_dict["normalizer.schema_version"])
            != self.normalizer.schema_version_value
            or not bool(state_dict["normalizer.fitted"])
            or float(state_dict["normalizer.epsilon"]) != self.normalizer.eps
            or bool(state_dict["projection_normalized"])
            != bool(self.projection_normalized)
        ):
            raise ValueError("Normalization schema/statistics mismatch")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class ThreeGroupCohortDecisions(CohortDecisions):
    observation_builder = None

    @torch.no_grad()
    def step(self, k, hidden, active):
        indices = torch.where(active)[0]
        for index in indices.tolist():
            self.histories[index].append(hidden[index].detach().clone())
        stopped = torch.zeros_like(active)
        if k % 10:
            return stopped
        if k == 80:
            return active.clone()
        observations = torch.stack(
            [
                self.observation_builder(torch.stack(self.histories[index]), k)
                for index in indices.tolist()
            ]
        )
        logits, values = self.model(observations)
        distribution = Categorical(logits=logits)
        actions = distribution.sample() if self.stochastic else logits.argmax(-1)
        t = k // 10 - 1
        self.x[indices, t] = observations
        self.actions[indices, t] = actions
        self.visited[indices, t] = True
        self.logprobs[indices, t] = distribution.log_prob(actions)
        self.values[indices, t] = values
        stopping = indices[actions == 0]
        stopped[stopping] = True
        self.budgets[stopping] = k
        return stopped


@contextmanager
def patched_decision_adapter(adapter):
    from scripts import run_pooled_controller_ppo as pooled
    from scripts import train_controller_from_profile as validation

    previous = pooled.CohortDecisions, validation.CohortDecisions
    try:
        pooled.CohortDecisions = validation.CohortDecisions = adapter
        yield
    finally:
        pooled.CohortDecisions, validation.CohortDecisions = previous


def load_three_group_checkpoint(path, controller_type, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"]
    width, dim = state["encoder.0.weight"].shape
    model = controller_type(
        dim,
        width,
        eps=float(state["normalizer.epsilon"]),
        projection_norm=bool(state["projection_normalized"]),
    )
    model.load_state_dict(state)
    return model.to(device).eval()
