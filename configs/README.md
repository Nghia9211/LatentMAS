# Configuration inventory

Cleaned on 2026-10-21: 23 YAML files reduced to 12.

## Current controller experiment

controller_ppo_cosine_state64_offline_v2.yaml is the latest cosine10,
64-iteration PPO configuration. Kept for diagnosis/retry; the v2 attempt failed
during engine initialization. This file does not indicate a running training job.

## Reusable method configurations

- baseline.yaml, latent_mas.yaml, text_mas.yaml: method entry points.
- controller_ppo_matched.yaml, controller_ppo_normalized.yaml,
  controller_ppo_uniform_explore.yaml, profile_medqa_pilot.yaml: existing script
  defaults, source-snapshot inputs, or profiling templates.
- benchmark_controller_warmstart.yaml, controller_ppo_train1024_baseline32.yaml,
  controller_ppo_train1024_no_warmstart32.yaml,
  controller_ppo_train1024_no_warmstart_no_tie32.yaml: configuration fixtures
  used by existing regression tests.

## Removed historical configurations

Obsolete benchmark/pilot versions, cosine v1, simple-state200 attempts,
completed warm-start/LR/duration ablations, and the old train1024 profiling
launch configuration have been removed. Historical reports may still name
their original paths.

An exact SHA256-verified backup and per-file manifest are outside the codebase:

/data/nghialt/.remote-workspace/jobs/latentmas-config-prune-20261021-v1/removed-configs.tar.gz

/data/nghialt/.remote-workspace/jobs/latentmas-config-prune-20261021-v1/manifest.json

Run artifacts, source snapshots, datasets, checkpoints, and historical logs
remain in place. Recover only the required archive member when needed;
historical output paths must not be reused for a new experiment.
