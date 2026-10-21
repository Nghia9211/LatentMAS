# Controller scripts

Core implementation lives in [controller/](../controller/README.md).
Select the entrypoint for the intended experiment;
these files are not a pipeline to execute in filename order.

## Profiling and data

| Task | Entrypoint |
|---|---|
| Export / verify official MedQA splits | `download_medqa_splits.py`, `verify_medqa_splits.py` |
| Collect / verify train-validation profiles | `profile_controller_pool.py`, `verify_controller_pool.py` |
| Collect repeated B responses from archived A | `profile_b_seed_repeats.py` |
| Schedule B repeats and required rest | `schedule_b_seed_repeats.py` |
| Migrate archived B-repeat plans with provenance | `migrate_b_seed_repeats.py` |
| Expand profiling to train1024 / supervise workers | `profile_controller_expansion.py`, `schedule_profile_expansion.py` |

## Training and evaluation

| Task | Entrypoint |
|---|---|
| Legacy normalized controller helpers | `train_normalized_controller.py` |
| Expanded-data identity/protocol helpers | `train_expanded_controller.py` |
| Expanded checkpoint / fixed-K evaluation | `evaluate_expanded_controller_test300.py` |
| Checkpoint evaluation | `evaluate_matched_ppo_checkpoint.py` |
| Saved-output loop audit | `audit_saved_loop_budgets.py` |

Evaluation script names are historical; confirm the configured cohort size and
source IDs in the generated plan, rather than inferring sample count from a name.

## Shared helpers retained for active callers

- `run_matched_controller_ppo.py`, `run_pooled_controller_ppo.py`,
  `train_controller_from_profile.py`: rollout, cache, checkpoint and training helpers.
- `run_ppo_with_wandb.py`, `validate_controller_wandb.py`: tracking and validation,
  including model-mode and RNG preservation.
- `profile_controller.py`, `verify_controller_profile.py`: profile/schema helpers
  and independent reference verification.
- `pilot_greedy_b.py`: archived-prefix reconstruction and decode/scoring helpers.

Warm-start/replay updates are in `controller/training.py`; historical cohort
selection is in `controller/matched.py`; fresh train-only selection is in

Retired benchmark, test16, L2-training, uniform-pilot and completed ablation CLIs
are removed. Shared helpers remain because current runners import them.

## Before launching

Read [publication/setup notes](../docs/README.md) and inspect the
selected YAML and `--help`. Historical server plans include absolute paths,
process identities, source hashes and scheduling gates. Preserve their original
bytes and create a new versioned plan for a new execution environment. Never
patch hashes or bypass gates to resume an old run.

The 2026-10-17 cleanup/migration notes describe their original snapshot. Later
profiling, training and evaluation artifacts exist on the server and are
intentionally excluded from Git. Check their completion markers and logs before
scheduling additional GPU work.

## Current fresh-rollout path

train_cosine_state_ppo.py uses controller/online_three_group.py. Task-owned runner files are thin imports;
PPO math and rollout implementation live in the repository.
An offline preset refers to W&B storage, not offline grid training.
See [the current State and PPO guide](../docs/README.md).
The scalar mirror runs separately with python -m controller.tracking_mirror.
