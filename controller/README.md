# Controller package

Các module lõi của adaptive latent handoff được gom ở đây. Các entrypoint
`scripts/*.py`, YAML trong `configs/`, và tests vẫn giữ vị trí cũ.

| Module | Vai trò |
|---|---|
| `profile.py` | Observation/phi4, trajectory collector, dataset và archive helpers |
| `matched.py` | Prompt/cohort và online rollout A → B |
| `sampler.py` | Train pool và shuffled sampler |
| `warmstart.py` | Encoder và supervised warm-start |
| `ppo.py` | Actor/critic, action sampling và PPO loss |
| `training.py` | Warm-start, replay-PPO and live PPO updates; optional per-step logging |
| `normalized.py` | Normalization và normalized controllers/checkpoint loader |
| `uniform_explore.py` | Uniform-budget initialization và tie-weight schedule |
| `tracking.py` | W&B tracking |
| `metrics.py` | Rollout/outcome diagnostics và phân bố correctness theo budget |
| `job_gate.py` | Process identity, release/rest gates; chỉ dùng standard library |
| `checkpoints.py` | Checkpoint selection from committed training iterations |
| `expanded_dataset.py` | Expanded train/validation profile loading and split checks |
| `server_training.py` | Server training, runtime reservations and source freezing |

Ví dụ import từ project root:

```python
from controller.ppo import PPOController
from controller.normalized import NormalizedPPOController
from controller.profile import raw_observation
```

Không còn các module `controller_*.py` ở project root. Package initializer
không import torch, vLLM hoặc W&B để scheduler vẫn chạy nhẹ trên CPU.
vLLM worker extension mới là `controller.profile.ProfileWorkerInspector`;
hai YAML đang dùng và config validator đã được cập nhật.

## Phạm vi refactor và dữ liệu lịch sử

Không đổi công thức observation/reward/loss, kiến trúc, weights, precision,
budget hay decoding. Checkpoints dùng state dict giữ nguyên tên tensor keys;
không hỗ trợ pickle nguyên object qua module path cũ.

`profiles/`, `evaluations/`, `source_snapshot/`, manifest/plan và checksums
lịch sử không được viết lại. Verifier đọc snapshot vẫn kiểm được archive cũ.
Source locks của job cũ cố ý không khớp working tree sau refactor: không tự
resume seed43/44 hoặc sửa hash cho qua kiểm tra. Cần migration/versioned plan
có provenance trước khi tiếp tục job cũ, cùng việc xử lý đường dẫn server.

Nguồn chụp cho run mới dùng `controller/*.py` và bao gồm `controller/__init__.py`.
Xem [workflow/scripts](../scripts/README.md) và [tài liệu controller](../docs/README.md).

## Cleanup entrypoints cũ

Các CLI benchmark warm-start/PPO, so timing một fresh iteration, greedy test16
và summary pilot32 đã bỏ. Phần còn dùng được giữ nguyên: `training.py`,
`matched.select_test16`/`require_train`, `sampler.fresh_train_items`,
`metrics.budget_patterns`. The retired benchmark fixtures were removed; maintained model/update tests use
controller.training directly.
CLI W&B wrapper còn hai lựa chọn `matched`/`pooled`, không còn `--runner fresh`.

## Current three-group controllers

- simple_state.py and cosine_state.py retain versioned public classes and observation formulas.
- three_group.py shares normalization, model plumbing, runtime decisions and loading.
- online_three_group.py owns fresh A/B rollout PPO for both schemas.
- tracking_mirror.py mirrors scalars with safe startup and JSONL handling.
- checkpoints.py dispatches same-sized L2/cosine states by their schema.

See [the current State and PPO guide](../docs/README.md).
Existing parameter/buffer names and checkpoint formats are preserved.

## Dependency-aware cleanup (2026-10-21)

All optimizer update helpers now live in training.py, including logged_ppo_update.
The separate ppo_logged.py module was removed; maintained callers and new-run
source lists use training.py. L2/cosine schema modules retain their public names,
exact observations, checkpoint keys and version guards, with consistent formatting.

The remaining historical modules are still imported by current scripts,
checkpoint dispatch or regression tests. They must not be removed based on age
alone. Old source snapshots/manifests are immutable; prepare a fresh versioned run
after source changes rather than modifying a historical source lock.
