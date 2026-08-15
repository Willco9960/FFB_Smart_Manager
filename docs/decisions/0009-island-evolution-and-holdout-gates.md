# ADR-009: Use island evolution with chronological holdout gates

## Status
Accepted

## Context

The modular trainer is sequential within one population, but its historical
scenario evaluations can run in parallel. A single 100-generation population
can still converge early, while raw training fitness can disagree with future
season performance.

## Decision

Add two separate controls:

1. `scripts.evaluate_modular_policy_holdout` evaluates a saved modular policy
   on a season that was not used during training and reports wins, points for,
   playoff rate, championship rate, and transaction reward.
2. `scripts.train_modular_policy_islands` runs independent policy populations
   in parallel for bounded segments. At each segment barrier, islands exchange
   one elite policy in a ring, preserving exploration while sharing progress.

Island workers use one scenario-evaluation worker each. This avoids nested
process oversubscription; the existing trainer remains the preferred path when
scenario-level parallelism is more appropriate.

## Alternatives Considered

### Assign contiguous generations to separate workers

Rejected because generation 11 requires the evolved population produced by
generation 10. Running those generations independently changes the algorithm.

### Run ten independent full experiments with no migration

Useful as a baseline, but it wastes information between runs. Island migration
provides controlled information exchange at explicit barriers.

### Select models only by training fitness

Rejected because it can reward historical overfitting. A chronological holdout
is required before calling a policy an improvement.

## Consequences

- Island training can use more CPU cores without violating generation order
  within an island.
- Segment barriers create natural checkpoints and inspection points.
- Island training has additional orchestration overhead and is not guaranteed
  to beat the standard trainer.
- Holdout evaluation adds time, but prevents selecting a policy that only wins
  on the training seasons.
