# Data and reproducibility

## Information cutoff

The simulator treats each decision as a historical information boundary. A feature is allowed only if a real manager could have observed it before that decision.

| Decision | Allowed information | Forbidden information |
| --- | --- | --- |
| 2021 draft | 2020 seasons, preseason projections, known roster/team context | 2021 weekly or final outcomes |
| Week N lineup | Prior weeks, current pre-game status, matchup context, projections available before lock | Week N final score or future weeks |
| Week N waiver | Prior production, current availability, known injuries/roles, pre-decision projections | Future breakout or final season totals |
| Week N trade | Current rosters, prior results, needs, projections, transaction rules | Later-week results used as if known at proposal time |

## Season windows

The long-running experiments commonly use 2001–2024 for training and 2025 as a holdout. The exact window belongs in each report. Older seasons may be used for lookback features, but their availability and schema must be audited first.

## Artifacts

- `data/raw/`: downloaded source files; ignored when large.
- `data/processed/`: normalized season files and caches.
- `data/models/`: model checkpoints and training checkpoints. Segmented vacation runs live under `data/models/vacation_runs/<run-id>/` and include an atomic `training_state.pt` plus `manifest.json`.
- `reports/`: charts, JSON evaluations, and human-readable summaries.
- `logs/`: terminal transcripts from long runs. Keep `Tee-Object` output under a run-specific subdirectory such as `logs/vacation/<run-id>/`.

Generated artifacts are evidence for an experiment, not replacements for the code and configuration that produced them.

## Identity and feature provenance

Player joins use the source `player_id` whenever available. Deterministic
fallback IDs are used only for legacy rows and team-defense aggregates. Draft
universes are built from the union of known pre-season and target-season
players; a `history_missing` mask identifies rookies or returning players
without prior-season history instead of silently dropping them.

Projection checkpoints include a `FeatureManifest` containing feature order,
schema version, decision cutoff, normalization statistics, source checksums,
identity-map version, and a digest. Loaders validate the digest and reject
incompatible feature names. Manager evaluation uses the versioned
`FitnessContract` so CPU and CUDA runs declare the same scoring and lineup
semantics.

## Required experiment record

Each meaningful run should record:

- Command and code revision.
- Training, validation, and holdout seasons.
- Feature cutoff and data source.
- Seed, population, generations, epochs, and mutation settings.
- Active lineup, waiver, trade, and playoff rules.
- Baselines and ablation arms.
- Runtime and hardware.
- Checkpoint and report paths.
- Any fallback or validation failure.

## Evaluation expectations

Use chronological validation whenever possible. Compare against simple baselines such as ADP/projection, random, genome, neural, and disabled-transaction arms. Report weekly wins, points for, playoff rate, championship rate, lineup efficiency, transaction value, and uncertainty—not only a single fitness number.

Promotion is gated, not merely reported: at least two aligned unseen-season
deltas must have a positive bootstrap interval and must not regress weekly wins.
A model that fails the gate remains an experiment artifact and is not promoted
to the flagship checkpoint.

## Synthetic seasons

Synthetic seasons can increase scenario diversity, but they must be generated from realistic distributions and evaluated separately from real seasons. A synthetic improvement is not evidence of real-season improvement until it survives a chronological real-season holdout.
