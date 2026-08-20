# Frontier Architecture Decisions

**Review date:** 2026-08-19  
**Scope:** CUDA manager-policy training, simulation validity, chronology, self-play, and RTX 3080 throughput.

## Decision

The project should remain a compact, rules-constrained manager policy trained by supervised behavior pretraining followed by evolutionary low-rank/adapter search and competitive evaluation. A larger neural architecture or a 1,000-generation run is not justified until the objective and holdout evidence support it.

## Evidence reviewed

- Current working-tree implementation and 347-test suite.
- Historical CPU/CUDA parity reports.
- Completed 400-generation and short calibration reports.
- Fresh final architecture calibration: 3 generations, population 16, four repeats, 2001-2023, self-play, 84.90 stable GPH.
- Fresh two-seed 2024/2025 holdout evaluation: candidate tied initial policy on every tested row; promotion false.
- Project AI engineering audit and data/reproducibility documentation.
- Independent cloud reviews of scientific validity and CUDA throughput.
- Local runtime: PyTorch 2.11.0+cu128, RTX 3080 compute capability 8.6, `torch.compile` available but unvalidated for this dynamic workload.

## Ranked decisions

### P0: objective validity before scale

1. Keep the versioned `FitnessContract` and CPU reference as the promotion authority.
2. Keep transaction-enabled CPU/CUDA behavioral parity as a hard promotion gate.
3. Use explicit fixed-action transition fixtures for backend parity; do not require different CPU and CUDA policy implementations to choose identical actions as a proxy for policy quality.
4. Require action-level exact-vs-batched parity for draft, lineup, waiver, trade, and playoffs before enabling a fast path for promotion evidence.
5. Keep transaction-disabled runs labeled draft/lineup ablations.

### P1: data and generalization

1. Preserve chronological 2000-2023 training eligibility and 2024/2025 holdouts.
2. When subsampling, retain both oldest and newest eras using deterministic endpoint-preserving selection.
3. Replay the full season window at a declared interval.
4. Evaluate at least two holdouts across multiple independent scenario seeds; never promote from one seed or one season.
5. Keep projection, legal baseline, initial policy, and competitive archive comparisons in every report.

### P1: search quality

1. Keep elite preservation, immigrant fraction, adapter-only mutation, and population-diversity monitoring.
2. Report action diversity by head in addition to parameter diversity before trusting a population.
3. Treat the opponent archive as a frozen snapshot store until ratings are updated from actual pairwise matches; do not interpret its ratings as calibrated Elo.
4. Use controlled ablations for mutation scope, population, archive size, scenario noise, and self-play cadence.

### P1: throughput

1. Keep vectorized lineup scoring and shared-policy team batching where exact action parity passes.
2. Keep exact per-policy evaluation as the audit fallback.
3. Profile the trade candidate expansion; prefer chunking/incremental deltas only after exact action parity and a real-data benchmark.
4. Do not enable `torch.compile` or CUDA Graphs by default. Dynamic rosters, masks, transaction branches, and changing scenario shapes make them experimental.
5. Report stable GPH, population evaluations/hour, and normalized scenario evaluations/hour. The current stable reference is approximately 84.90 GPH, not the earlier 98-GPH peak.

## Latest parity evidence

The post-contract historical transaction comparison for 2023/256 players still reports:

```text
exact standings: false
exact champion: false
exact weekly scores: false
maximum weekly score delta: 66.86
```

This proves deterministic tie-breaking is not the root cause of the full mismatch. The CPU and CUDA transaction/objective implementations still differ structurally. Transaction-enabled CUDA remains exploratory and promotion-blocked.

## Latest self-play corrections

The routed self-play evaluator now evaluates each candidate across every team
assignment using the same scenario bank and aggregates the balanced results.
Candidate auxiliary gains are also team-attributed: lineup, waiver, and trade
rewards from opponent-controlled teams are no longer credited to the candidate.
A regression fixture covers non-monotonic two-team attribution.

This correction is expected to reduce self-play generations/hour because it runs
one season per candidate-team assignment. The short CUDA smoke path completed
successfully, but its one-generation rate is not a production benchmark.

## Latest contract safety correction

CUDA self-play now preserves candidate-only auxiliary attribution and balances
candidate team assignment across the league. Promotion-oriented transaction
training also fails closed when the CUDA evaluator does not consume the declared
replacement-value or invalid-action contract terms. Exploratory transaction runs
remain available, but they are explicitly diagnostic and cannot satisfy promotion.

The remaining transaction parity work is behavioral: CPU and CUDA still choose and
apply different dynamic waiver/trade actions on historical seasons. Exact action,
roster, ownership, weekly-score, standings, and playoff parity is still required
before enabling the full contract.

## Required training protocol

1. Run preflight over the exact chronological data window.
2. Run a short transaction-enabled calibration and a matched transactions-disabled control.
3. Run exact-vs-batched action parity on production-like fixtures.
4. Run a multi-seed holdout matrix on frozen candidates.
5. Start a long run only if validity gates pass and the experiment has one declared scientific question.
6. Preserve unique model, population checkpoint, report, and log artifacts.

## Explicit non-decisions

- Do not increase hidden size merely because the current model ties its initial checkpoint.
- Do not claim a transactions-disabled model is a complete manager.
- Do not use a short calibration's fitness trajectory as generalization evidence.
- Do not optimize generations/hour by silently lowering population, repeats, players, self-play, or holdout quality.
