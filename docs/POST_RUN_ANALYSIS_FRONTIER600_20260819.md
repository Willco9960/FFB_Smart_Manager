# Post-Run Analysis: Frontier 600

Date: 2026-08-19  
Run: `frontier_2001_2023_overnight_20260819_2220_frontier600_validated`

## Execution result

The run completed normally through generation 600.

- Training seasons: 2001–2023
- Holdouts: 2024 and 2025
- Population: 16
- Scenario repeats: 4
- Self-play: every generation
- Opponent archive: 64
- Policy heads: flattened/batched CUDA route
- Transactions: disabled
- Deterministic CUDA: disabled; TF32 enabled
- Elapsed time: 22,749.48 seconds / 6.32 hours
- Steady-state throughput: approximately 94.95 GPH
- Scenario evaluations: 883,200
- Model: `data/models/frontier_2001_2023_overnight_20260819_2220_frontier600_validated.pt`
- Full checkpoint: `data/models/frontier_2001_2023_overnight_20260819_2220_frontier600_validated_state.pt`
- Durable log: `logs/frontier_20260819_2220_frontier600_validated.log`

The model and full checkpoint are preserved and must not be overwritten by follow-up experiments.

## Optimization trajectory

The final generation was not the strongest risk-adjusted generation.

- First-generation risk-adjusted fitness: 242.03
- Maximum risk-adjusted fitness: 269.12
- Final risk-adjusted fitness: 238.21
- Final minus best risk-adjusted fitness: -30.91
- Final population diversity: 184.14
- Last-20-generation mean diversity: 189.54

This indicates noisy evolutionary search and/or selection pressure that did not preserve the best measured trajectory as the final reported population metric. The saved `best_policy` artifact is the candidate used for holdout evaluation, but reports must expose best-versus-final explicitly.

## Holdout matrix

| Season | Seed | Candidate fitness | Initial fitness | Delta | Risk-adjusted delta |
|---:|---:|---:|---:|---:|---:|
| 2024 | 1001 | 400.14 | 355.08 | +45.06 | +43.00 |
| 2024 | 2001 | 467.58 | 409.37 | +58.21 | +60.56 |
| 2025 | 1001 | 290.50 | 290.50 | 0.00 | 0.00 |
| 2025 | 2001 | 379.34 | 379.76 | -0.42 | -0.44 |

Per-season multi-seed results:

- 2024: positive under every seed.
- 2025: not positive under every seed; mean raw delta -0.21 and mean risk-adjusted delta -0.22.
- Multi-seed promotion readiness: `false`.

The single-run report also shows that the candidate loses to the legal projection baseline on 2025:

- Candidate: 283.17
- Projection baseline: 350.03
- Candidate delta: -66.86

## Validity classification

| Axis | Result |
|---|---|
| Execution | Complete and artifact-backed |
| Optimization | Positive peaks, but noisy and regresses by final generation |
| Generalization | Mixed: strong 2024, neutral/slightly negative 2025 |
| CPU/CUDA validity | No-transaction parity is covered; transaction parity remains unresolved |
| Promotion | Blocked |

This is a high-throughput, transactions-disabled exploratory draft. It is not a promotion-ready full-manager policy.

## Improvement ledger

### Fixed in this post-run pass

1. **Run provenance capture**
   - `_git_identity()` now resolves the repository explicitly with `git -C`.
   - Working-tree status is included in the diff digest input.
   - Direct verification now returns the current revision and a non-empty digest.

2. **Warm-up distortion in throughput summaries**
   - `summarize_cuda_throughput()` now accepts `warmup_generations`.
   - The training report excludes the first five generations from the stable-rate range/mean while preserving total elapsed and aggregate GPH.
   - The report records `warmup_generations_excluded`.

3. **Best-versus-final ambiguity**
   - Completed reports now include `optimization_summary` with the best risk-adjusted generation, final generation, and final-minus-best delta.

4. **Regression coverage**
   - Added a warm-up throughput regression test.
   - Existing focused training/promotion tests still pass.

### Required but not silently changed

5. **2025 generalization failure**
   - Requires a new controlled ablation; it must not be tuned against the holdout.
   - Candidate should be evaluated with training-only walk-forward validation or a reserved validation era before selecting mutation/risk/scenario changes.

6. **Transaction-enabled parity**
   - CPU and CUDA still use different transaction state machines and policies.
   - Transactions remain disabled until action, state, weekly-score, standings, playoff, and reward traces match exactly.

7. **Complete transaction fitness contract**
   - Replacement-value weighting and invalid-action penalties are not yet equivalent on CUDA.
   - Promotion-oriented transaction training must continue to fail closed.

8. **Deterministic qualification evidence**
   - The overnight artifact intentionally used the faster non-deterministic/TF32 route.
   - A separate deterministic qualification/resume smoke remains required before calling a result reproducible at the bitwise level.

9. **Process-tree cleanup**
   - Manual termination of wrapper shells previously left child Python training processes alive.
   - Future launch tooling should use a process-tree/job-object cleanup path and verify no child training processes remain after cancellation.

10. **Competitive baseline gap**
    - The candidate beats the initial policy on the 2024/2025 single-run report but loses to the projection baseline on 2025.
    - This is a model/objective result, not a reporting defect; it blocks promotion until a controlled follow-up improves the legal baseline comparison without holdout tuning.

## Follow-up order

1. Preserve this artifact set unchanged.
2. Land the provenance, warm-up, and optimization-summary fixes with tests.
3. Implement canonical CPU transaction traces and CUDA replay.
4. Re-establish exact transaction parity across historical seasons.
5. Add a training-only validation fold and controlled mutation/risk/scenario ablations.
6. Run independent multi-seed chronological holdouts only after the candidate/configuration is frozen.
7. Require all holdouts to beat the initial policy and legal projection baseline before promotion.
