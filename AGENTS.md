# Repository working instructions

## Remote boundary

Work in /data/nghialt/LatentMAS over the configured SSH workflow.
Keep task files under /data/nghialt. Read the remote-workspace skill/profile.
Do not signal another user's processes or alter immutable experiment snapshots.
Preserve unrelated work and the existing Git index; review before staging.

## Documentation policy requested by the user, 2026-10-21

- docs/README.md is the single canonical research/operation document.
- The old per-run, migration, proposal, pilot and abandoned-analysis documents
  have been consolidated and removed from the working tree.
- Do not recreate, restore, expand, or resume those discarded documentation or
  experiment branches unless the user explicitly asks to revisit them.
- Routine training, status checks, code fixes, and new experiment outputs must
  use task logs/manifests. Do not automatically edit or add research docs.
- Edit the canonical doc only when the user explicitly requests documentation
  or analysis updates. A later explicit user request takes precedence.
- Keep raw results, manifests, logs and checkpoints truthful and immutable;
  dropping a weak result from the active narrative does not authorize deleting
  evidence or claiming an unproven improvement.
- Current research direction is the causal h_k + cosine distance over 10 steps
  + k/80 controller with fresh rollout PPO. Old full-state/L2/grid/pilot records
  are historical references, not automatic instructions to resume them.
