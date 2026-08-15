# ADR-013: Isolate the CUDA simulator prototype behind a parity gate

## Status

Accepted

## Context

The full-season simulator currently uses Python objects and irregular draft,
waiver, trade, lineup, and playoff rules. Replacing it wholesale with CUDA
would be expensive and could silently change fantasy outcomes. The RTX 3080
may still provide a large speedup for the dense, batched portions of the
simulation, so the project needs a measurable experiment without risking the
working CPU path.

## Decision

Add `gpu_sim` as an experimental, non-production backend. Its first scope is a
tensorized projection-best snake draft and offensive lineup scoring. It must:

- match the CPU reference for deterministic draft and lineup fixtures;
- run on CPU when CUDA is unavailable;
- report elapsed time and generations per hour;
- report both batch throughput and scenario throughput so prototype rates are
  not confused with full-season trainer GPH;
- remain separate from `fantasy_engine` and the full-season trainer;
- only become a production backend after output parity and a measured speedup.

The migration groundwork keeps scenario data in a `TensorScenarioBatch`, reuses
one in-place draft score buffer, and provides an opt-in synchronized stage
profiler. Normal benchmark runs do not pay profiler synchronization overhead.

Waivers, trades, playoffs, and real manager-policy competition remain in the
existing CPU simulator until each receives its own parity test. The new
full-season tensor path is therefore a benchmark and migration target, not a
drop-in replacement.

## Alternatives considered

### Replace the simulator with CUDA immediately

Rejected because a failed rewrite could invalidate historical results and
remove the current fallback.

### Move only neural-network inference to CUDA

Useful but insufficient: inference is not the dominant cost in the current
Python-heavy full-season simulation.

### Keep the CPU simulator only

Safe but does not test the potential throughput gain from batched tensor state.

## Consequences

The repository now has a safe benchmark for the GPU upper-bound experiment.
The prototype does not yet claim to accelerate the complete manager simulator.
The first profiled CUDA smoke run found lineup scoring to be the larger kernel
stage, so future optimization should target lineup selection before expanding
to transaction logic.

The full-stage smoke benchmark also establishes a batching requirement: small
batches of 8 leagues remain CPU-faster because CUDA launch overhead dominates,
while a 256-league batch is GPU-faster. Production integration should tile
population members and historical scenarios into a sufficiently large batch
before dispatching a CUDA season kernel.

The full-stage prototype now covers weekly standings (wins, losses, ties,
points-for, and points-against), the ESPN six-team playoff bracket, and
transaction counters. Waiver and trade value attribution is not yet equivalent
to the CPU replay reward pipeline, so those counters are diagnostic only until
transaction-level parity is accepted.

The historical adapter and comparison runner provide a safe overnight gate:
without transactions, real seasons use the previous season for draft
projections and must match CPU standings and champion exactly. With transactions
enabled, the report is explicitly an outcome delta because tensorized waiver
and trade proposals are not yet action-identical to the CPU agents.
Benchmark reports must be compared against the CPU reference before expanding
scope.
