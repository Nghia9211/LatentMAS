# Configuration inventory

The maintained directory contains 12 YAML templates after cleanup on 2026-10-27.
Completed run configurations and snapshots remain with server jobs.

[Verified results and current job states](../docs/README.md#verified-results)
were refreshed on 2026-10-27. Cosine v8 completed 64 iterations and test500
at 69.6%; the retained v2 template below is historical. Native KVComm completed;
the native adaptive campaign was canceled, and K2 shortcut has not trained.
Do not infer execution status or quality from a YAML filename.

## Historical controller template

controller_ppo_cosine_state64_offline_v2.yaml is retained as the documented
cosine10, 64-iteration PPO template. Its v2 attempt failed at engine initialization;
it is not the latest completed run or evidence of an active job. The historical
v3-v8 retry presets were removed on 2026-10-27. Exact run configurations remain
with their historical jobs; new output/job paths must be used for a new run.
These historical controller templates use the original hybrid protocol and must
not be launched as native HF direct-KV experiments without adaptation and checks.

## Reusable method configurations

- baseline.yaml, latent_mas.yaml, text_mas.yaml: method entry points.
- controller_ppo_matched.yaml, controller_ppo_normalized.yaml,
  controller_ppo_uniform_explore.yaml, profile_medqa_pilot.yaml: existing script
  defaults, source-snapshot inputs, or profiling templates.
- benchmark_controller_warmstart.yaml, controller_ppo_train1024_baseline32.yaml,
  controller_ppo_train1024_no_warmstart32.yaml,
  controller_ppo_train1024_no_warmstart_no_tie32.yaml: configuration fixtures
  used by the server-only regression tests; tests/ is excluded from GitHub.

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

The 2026-10-27 cleanup backup and per-file reasons are recorded outside the repo:
/data/nghialt/.remote-workspace/jobs/latentmas-code-cleanup-20261027-v1/cleanup_manifest.json.
