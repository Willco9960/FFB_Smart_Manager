# ADR-012: Optimize Petic GPH with cached scenario workers

## Status
Accepted

## Context

The modular self-play loop is dominated by CPU historical simulation. Before
this change, every scenario evaluation task re-serialized the full league and
weekly performance tables for every generation. That repeated transfer work
limited useful CPU utilization even when multiple workers were enabled.

The project needs a measurable throughput target: **Petic GPH** (generations
per hour). Quality gates must remain unchanged while increasing this number.

## Decision

- Load the historical scenario library once in each persistent scenario worker.
- Send policy state and a scenario index for subsequent generation jobs instead
  of re-sending the season data.
- Default historical evaluation workers to eight on the target desktop, while
  preserving `--evaluation-workers 1` for deterministic debugging.
- Retain only the top 64 generation candidates in resumable checkpoints; the
  final audit requires at most eight candidates.
- Skip expensive final candidate audits on intermediate vacation segments and
  perform the audit only on the final segment.
- Emit `generations_per_hour` in generation telemetry and logs.

## Alternatives Considered

### Move the complete simulator to CUDA

Rejected because the simulator is Python/control-flow heavy and its neural
decisions are small. CUDA transfer overhead would exceed the useful matrix
work for individual decisions.

### Spawn more workers without caching

Rejected because it increases CPU and memory pressure while still repeating
large season-object serialization.

### Remove checkpoints

Rejected because resumability is a project requirement. Candidate archiving
reduces checkpoint cost without sacrificing generation-boundary recovery.

## Consequences

- Workers consume more resident memory because each process retains the season
  cache; eight workers remain appropriate for the 32 GB development machine.
- A two-season smoke benchmark improved self-play throughput from 44.5 to 71.9
  generations/hour with two cached workers and to 78.8 generations/hour with
  four workers. Full-run throughput still depends on scenario count and
  population size.
- Petic GPH is now visible in logs, while holdout and final-evaluation metrics
  remain the authority for model quality.
