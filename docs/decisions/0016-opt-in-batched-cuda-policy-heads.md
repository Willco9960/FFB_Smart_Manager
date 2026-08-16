# ADR-016: Opt-in batched CUDA policy heads

## Status

Accepted

## Date

2026-08-16

## Context

The full CUDA season simulator already keeps draft, lineup, waiver, trade, and
playoff state on the device. Its conservative population evaluator ran each
policy independently so every state-conditioned head could be audited. That
protected correctness but left GPU throughput on the table.

The flattened evaluator can represent each candidate policy as a contiguous
scenario block and use a `torch.func.vmap` ensemble. The risk is silently
mixing policy identity with team identity during in-season decisions.

## Decision

Add an explicit `--batched-policy-heads` mode to
`scripts.train_cuda_manager_policy`. It routes the complete season, including
transactions, through the flattened CUDA population evaluator. Keep exact
per-policy head evaluation as the default until larger historical parity sweeps
justify changing the default.

The batched route must pass a transaction-enabled metric parity test against
the exact evaluator before it is used for a long run. Every run records its
routing mode in the terminal output and report.

## Alternatives considered

### Silently switch every CUDA run to batching

Rejected: a throughput optimization must not hide a change in action routing.

### Keep batching draft-only

Rejected: draft-only speed does not solve the full-season bottleneck; lineup,
waiver, and trade heads must be included in the measured path.

### Continue exact sequential evaluation only

Rejected: it is the correctness baseline, but it prevents the RTX 3080 from
processing independent candidate-policy state blocks concurrently.

## Consequences

- CUDA training can explicitly trade a validated routing assumption for higher
  population throughput.
- Exact evaluation remains available for debugging and promotion audits.
- The transaction-enabled parity test protects the current policy-head shape,
  while historical CPU/CUDA parity remains the final promotion gate.
- Measured GPH must be reported separately for exact and batched modes; a small
  synthetic benchmark is not evidence of full historical throughput.
