# ADR-014: Shared transitions and pre-training gates

## Status

Accepted

## Date

2026-08-16

## Context

The project has CPU and CUDA simulators, four manager heads, replay records,
and long-running evolutionary training. They previously had enough shared
rules to score seasons, but not one explicit state/action/transition contract.
Long runs could also begin with incomplete source columns, uninitialized
manager heads, or uncertainty outputs that never reached policy features.

Dynamic CPU/CUDA transaction policies can choose different legal actions. A
parity report must not turn that expected behavioral difference into a false
claim of backend equality.

## Decision

Adopt five gates before a flagship training run:

1. Serialize `ManagerState`, `LegalActionMask`, `ManagerAction`, and
   `ManagerTransition` with a versioned fitness-contract digest.
2. Test CPU/CUDA transaction execution with identical explicit waiver/trade
   fixtures. Keep dynamic policy comparisons diagnostic until selected actions
   are identical.
3. Require a `DataAvailabilityManifest` for every chronological season. Core
   columns need at least 99% observed coverage; optional context is recorded
   and masked rather than fabricated.
4. Feed projection floor, median, ceiling, and boom probability into the
   policy state and checkpoint feature manifest.
5. Behavior-pretrain draft, lineup, waiver, and trade heads, then run a
   finite-loss/contract/synthetic-season preflight before promotion.

## Alternatives considered

### Start long runs without a preflight

Rejected: a cheap structural failure should never consume an overnight run.

### Treat dynamic CPU/CUDA transactions as exact parity

Rejected: different policies can make different legal choices. The report must
surface that difference rather than hiding it.

### Require 100% nonblank coverage for every source field

Rejected: nflverse contains a small number of documented blank identity fields
on stat-only rows. A 99% core threshold remains fail-closed without rejecting
valid source files; manifests preserve the exact coverage for review.

### Keep uncertainty only in projection reports

Rejected: a manager cannot reason about risk if the policy state receives only a
point estimate.

## Consequences

- CPU, CUDA, and replay tooling can compare transition digests.
- Long runs fail quickly when source data or policy heads are incomplete.
- The transaction fixture gate is exact and reproducible, while dynamic action
  differences remain visible and honest.
- Teacher pretraining is a warm start, not a claim that the teacher is optimal;
  self-play and chronological holdouts still determine promotion.
- Old checkpoints remain loadable when they lack the newer feature manifest;
  new checkpoints record the expanded state schema.
