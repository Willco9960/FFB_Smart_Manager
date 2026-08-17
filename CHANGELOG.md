# Changelog

## Unreleased

### Changed
- Hardened CUDA manager training: batched population evaluation now preserves
  the active fitness contract, lineup rules, and distributional projection
  context; custom contracts are propagated safely; candidate-only routing
  prevents opponent control; redundant opponent draft and in-season policy
  inference is skipped; playoff pairing uses standings seed ranks; training
  rejects invalid chronological holdout windows; and resumable CUDA checkpoints
  are replaced atomically.
- Promotion gates now require at least two aligned unseen seasons and validate
  season provenance, matching the documented chronological holdout policy.
- Historical player pools now exclude target-season-only players by default and
  retain projection-season identity/team metadata.
- Added an isolated tensorized draft and lineup CUDA benchmark with CPU parity tests; the production simulator remains unchanged.
- Added a reproducible CPU-versus-CUDA comparison report for the tensorized prototype.
- Added resident tensor scenario batches, in-place draft scoring, and opt-in CUDA stage profiling for GPH optimization.
- Added benchmark-only tensorized full-season stages for weekly scoring, waivers, trades, standings, and playoffs; large-batch crossover measurements are documented.
- Added baseline-relative robust selection for modular self-play.
- Added warm-start immigrant policies and stronger diversity-collapse detection.
- Expanded final candidate evaluation from three to eight policies.
- Added selection and opponent-relative fitness telemetry to generation logs.
- Added cached scenario data in persistent workers, bounded candidate archives,
  final-segment-only audits, and Petic GPH telemetry.

- Added atomic modular training-state checkpoints with population, RNG, candidate, metadata, and generation-cursor restoration.
- Added direct resume support through `scripts.resume_modular_manager_policy`.
- Added restart-safe segmented vacation training with manifests and per-run artifact directories.
- Added `scripts.show_modular_run_status` for non-invasive progress inspection.
- Preserved generation records across interrupted segment retries and made selected transaction mode explicit in final evaluation reports.
- Made new vacation runs default to the empirically stronger genome transaction strategy; neural/hybrid transaction training is now opt-in.
- Added bounded process-parallel historical scenario evaluation to use available CPU cores for the simulator while preserving deterministic agent ordering.
- Added modular-policy chronological holdout evaluation and an opt-in island trainer with segment-barrier elite migration.
- Added a controlled standard-versus-island trainer benchmark command.
- Reused scenario worker processes across generations, capped worker-side PyTorch threads, batched neural lineup scoring, and added automatic CUDA for batched pretraining.
- Added focused coverage for state round-tripping and manifest writes.

This project is experimental. Entries describe implemented behavior and research infrastructure; they do not imply that a policy is optimal in future leagues.

## Unreleased

### Added

- Public documentation structure covering architecture, data reproducibility, security, contribution, and decisions.
- Explicit 100% Spec-Driven Development policy and acceptance-criteria workflow.
- Explicit transparency notes for offensive-only scoring, placeholder UI state, experimental policies, and human-approved platform actions.
- Public contributor and coding-agent guidance.

### Documented

- Empirical modular-training runtimes: 6.99 hours for the latest bounded profile and roughly 14.2 hours for two earlier heavier runs.
- Current limitation that modular checkpoints are not yet full resumable training-state checkpoints.

### Existing research capabilities

- Historical draft and weekly-season simulation.
- Evolutionary and neural policy experiments.
- Walk-forward and multi-season evaluation commands.
- Transaction replay records for waiver and trade value analysis.
- Optional local language-model explanation layer.
- Offline UI prototype.
