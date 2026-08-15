# ADR-010: Use a hybrid CPU/CUDA execution plan

## Status
Accepted

## Context

The simulator performs many small Python operations for drafts, lineups,
waivers, trades, and matchups. The modular policy network is intentionally
small, so launching CUDA work for every individual decision can cost more than
the matrix operation itself. The behavioral-cloning, replay, and transaction
value stages instead operate on large batches of tensors.

## Decision

- Keep historical season simulation on bounded CPU worker processes.
- Reuse scenario workers across generations and cap PyTorch/BLAS threads inside
  each worker to prevent oversubscription.
- Use CUDA automatically for batched pretraining when a CUDA-enabled PyTorch
  build is available. Move policies back to CPU before process-based self-play.
- Batch neural lineup scoring and cache per-pick draft anchor counts to reduce
  repeated Python work.

## Alternatives Considered

### Put every simulator worker on the GPU

Rejected because the model is small and workers would contend for one GPU while
paying CPU-to-GPU transfer costs for tiny tensors.

### Use CPU only

Rejected because the pretraining stages already produce contiguous batches that
can use the RTX 3080 efficiently.

## Consequences

- `--training-device auto` uses CUDA for pretraining and CPU for simulation.
- `--training-device cpu` remains available for debugging and reproducibility.
- Runtime depends on scenario count, worker count, and process startup overhead;
  benchmark before committing to a large island run.
