# ADR-0015: Use common-random, risk-aware, resumable CUDA evolution

## Status

Accepted

## Context

The CUDA policy trainer is fast enough for overnight runs, but raw throughput
alone does not guarantee useful learning. Candidate policies must be compared
on the same randomized historical scenarios, noisy winners should not dominate
selection, and an interrupted run must not discard the entire evolutionary
population. The draft feature builder also contained a quadratic positional
ranking operation that became expensive at every pick.

## Decision

The flagship CUDA training path will:

- materialize a common scenario bank once per run and fork mutable states for
  each policy;
- rank candidates with a risk-adjusted fitness (`mean - penalty * stddev`);
- keep a chronological holdout season outside selection;
- checkpoint the full next-generation population, best policy, metrics, and
  Python RNG state after every generation;
- resume from that checkpoint with `--resume`;
- compute positional ranks with argsort/scatter instead of a dense pairwise
  player-by-player comparison;
- evaluate the population with a `torch.func.vmap` functional ensemble on
  CUDA, with sequential evaluation retained as a debug fallback;
- retain eager CUDA as the default when `torch.compile` cannot find Triton.

## Alternatives considered

### Independent random scenarios for each policy

Rejected because selection would confound manager quality with luck. Common
random numbers make policy differences more measurable while the holdout audit
still checks generalization.

### Keep only the best-policy checkpoint

Rejected because an interruption would lose the population diversity required
for evolutionary search and force a restart.

### Dense pairwise rank features

Rejected because its quadratic temporary tensor scales poorly with player-pool
size and repeated draft picks. Argsort/scatter preserves the rank feature with
lower memory traffic.

### Require `torch.compile`

Rejected for this Windows environment because the installed PyTorch build does
not include a working Triton runtime. Eager CUDA with AMP and TF32 is the
portable production path; compilation remains an optional benchmark.

### Evaluate one policy at a time

Rejected as the production CUDA path because the season kernel is dominated by
many small draft and lineup launches. Flattening the population into one
scenario batch increases GPU occupancy while the parity test guards against
changing policy scores.

## Consequences

- Runs are more reproducible and rankings are less noisy.
- Resume checkpoints are larger than best-policy files but remain small
  compared with historical tensor data.
- Scenario caching uses additional VRAM/RAM, bounded by the configured player
  count and repeat count.
- The CUDA trainer still uses tensorized baseline transaction behavior; full
  neural waiver/trade heads remain a separate parity milestone.
