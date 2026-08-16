# ADR-0014: Use a CUDA policy-conditioned season trainer

## Status

Accepted

## Context

The CUDA simulator could already execute many tensorized seasons per hour, but
its draft kernel selected players by projection only.  That measured simulation
throughput, not manager learning.  The CPU trainer remained the only path that
evaluated neural policies against competing teams, making overnight runs too
slow on the available RTX 3080.

## Decision

Add a separate CUDA evolutionary trainer that:

- loads leakage-safe historical seasons;
- lets one candidate neural policy draft against nine projection-best baseline
  teams;
- rotates the candidate's draft slot across replicated scenarios;
- keeps CUDA waiver/trade simulation enabled for the full-season fitness score;
- evolves policy parameters with selection, crossover, and mutation;
- writes a checkpoint and metrics after every generation.

The CPU object-oriented trainer remains available as the behavioral reference.
The CUDA trainer is an optimization path and must be compared against CPU
holdout results before it is treated as behaviorally equivalent.

## Alternatives considered

### Keep using the CUDA stress benchmark

Rejected because it measures projection-greedy simulation throughput and cannot
improve a manager policy.

### Run the existing neural trainer with CUDA only for gradient updates

Rejected as the immediate overnight solution because historical season
evaluation still dominates runtime on the CPU.

### Rewrite every transaction rule before enabling CUDA training

Deferred.  The tensorized waiver/trade stages are already usable for a first
policy-learning benchmark; action-by-action parity remains a separate gate.

## Consequences

- Overnight training can now use the RTX 3080 for both policy scoring and
  tensorized season simulation.
- The initial CUDA policy head controls draft decisions; waiver and trade
  decisions currently use the existing tensorized baseline logic.
- Candidate-vs-baseline evaluation is more informative than having every team
  use the same policy.
- CUDA and CPU results may differ when transaction approximations are enabled;
  reports must preserve that distinction.
