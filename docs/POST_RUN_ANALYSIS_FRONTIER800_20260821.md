# Frontier 800 Exploratory Run — Post-Run Analysis

Date: 2026-08-21
Run ID: `frontier_2001_2023_overnight_20260821_batched_800`

## Execution

The training workload completed all 800 generations and wrote the model, resumable
state checkpoint, and JSON report. The PowerShell wrapper returned exit code 1 after
training because Windows exposed a null child exit code after the child had already
been reaped. The report itself had `status: complete`, so the experiment artifacts
are valid and preserved.

The launcher was patched to treat a completed report as success when the child exit
code is null, while still failing when a nonzero exit code is explicitly available.

## Configuration

- Device: NVIDIA GeForce RTX 3080 / CUDA
- Training seasons: 2001–2023
- Holdouts: 2024 and 2025
- Population: 16
- Generations: 800
- Scenario repeats: 4
- Scenario refresh: every 25 generations
- Players: 256
- Self-play: enabled every generation
- Opponent archive: 64
- Full policy mutation: enabled
- Batched CUDA policy heads: enabled
- Deterministic mode: disabled
- TF32: enabled
- Transactions: disabled; this remains an exploratory draft-only ablation

## Throughput

- Total elapsed: 30,003.15 seconds / 8.33 hours
- Overall throughput: 95.99 generations/hour
- Stable throughput: 96.16 generations/hour
- Stable range: 83.13–103.83 generations/hour
- Population evaluations: 294,400
- Scenario evaluations: 1,177,600
- Normalized scenario evaluations/hour: 141,543.79

The prior slow launch reached only approximately 13 GPH because it silently used
exact per-policy heads and deterministic execution. The corrected launcher now makes
both controls explicit.

## Optimization trajectory

- Generation 1 risk-adjusted fitness: 242.03
- Best risk-adjusted fitness: 272.11 at generation 279
- Final generation risk-adjusted fitness: 236.17
- Final minus best: -35.94

The trainer correctly retained the global best policy rather than replacing it with
the final-generation policy. Future reports should continue exposing both best and
final values because the final population can regress after the best discovery.

## Chronological holdouts

### 2024

- Candidate: 383.14
- Initial policy: 424.68
- Projection baseline: 196.03
- Candidate risk-adjusted delta versus initial: -39.31
- Candidate risk-adjusted delta versus projection baseline: +184.09

### 2025

- Candidate: 325.39
- Initial policy: 280.44
- Projection baseline: 350.03
- Candidate risk-adjusted delta versus initial: +40.81
- Candidate risk-adjusted delta versus projection baseline: -22.13

This is mixed generalization. The candidate does not beat the initial policy and legal
projection baseline on every required holdout, so promotion remains blocked.
Transactions were disabled and historical CPU/CUDA realized reward parity remains
incomplete, which independently blocks promotion.

## Research-informed conclusions

The post-run findings agree with current PyTorch and NVIDIA guidance:

1. Measure the exact workload rather than assuming GPU utilization or compiler
   settings improve throughput.
2. Use flattened/batched policy-head execution for this production-shaped workload;
   it improved measured throughput from roughly 13 GPH to approximately 96 GPH.
3. Keep deterministic execution available for qualification and reproducibility
   checks, but do not silently use it for throughput-optimized exploratory runs;
   PyTorch documents that deterministic algorithms can be slower.
4. Do not enable `torch.compile` or CUDA graphs in the next production configuration
   without a same-workload warm-up benchmark, steady-state comparison, numerical
   parity check, and rollback condition.
5. Preserve chronological holdouts and the legal baseline. The run's mixed result
   is evidence for a controlled follow-up ablation, not a promotion claim.

Primary references consulted:

- PyTorch Performance Tuning Guide:
  https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- PyTorch `torch.compile` tutorial:
  https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- PyTorch reproducibility notes:
  https://docs.pytorch.org/docs/stable/notes/randomness.html
- NVIDIA CUDA Best Practices Guide:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

## Artifacts

- Report:
  `reports/frontier_2001_2023_overnight_frontier_2001_2023_overnight_20260821_batched_800.json`
- Model:
  `data/models/frontier_2001_2023_overnight_frontier_2001_2023_overnight_20260821_batched_800.pt`
- Resumable checkpoint:
  `data/models/frontier_2001_2023_overnight_frontier_2001_2023_overnight_20260821_batched_800_state.pt`

Promotion status: **blocked**.
