# ADR-008: Parallelize historical scenario evaluation

- Status: Accepted
- Date: 2026-08-13

## Context

The RTX 3080 was underused during long runs because most runtime was spent in Python draft, lineup, waiver, trade, and matchup simulation. Neural policy calls are small and frequent, so moving individual calls to a GPU does not solve the dominant bottleneck. Historical scenarios are independent within a generation.

## Decision

Evaluate historical scenarios in a bounded `ProcessPoolExecutor`. Each worker receives one scenario and returns results in the caller's agent order. The CLI exposes `--evaluation-workers`, defaulting to four, and `1` remains available for debugging or constrained machines.

## Consequences

- Python simulation can use multiple CPU cores instead of one serial path.
- Results remain comparable because scenario seeds and agent ordering are preserved.
- Process startup and model serialization add overhead for tiny runs, so this is intended for multi-scenario training workloads.
- The GPU remains available for genuinely large batched neural training, but it is not assumed to accelerate the simulator automatically.
