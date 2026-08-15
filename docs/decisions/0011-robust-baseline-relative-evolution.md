# ADR-011: Select policies with robust opponent-relative fitness

## Status
Accepted

## Context

The 200-generation `vacation_2026_03` run completed in 11.62 hours, but its
training and full-evaluation rankings diverged. The self-play loop selected
policies from rotating historical samples, while the final gate evaluated only
a small number of candidates across all seasons. Population diversity also
fell to roughly 0.004, indicating that the population was converging too
quickly around similar policies.

Absolute fantasy fitness is still important, but historical scenarios vary in
difficulty. A policy should receive credit when it beats the baseline managers
in the same scenario, not only when the scenario produces a large raw score.

## Decision

- Blend risk-adjusted absolute fitness with risk-adjusted fitness relative to
  the same-scenario baseline opponents.
- Keep a larger final candidate audit by evaluating eight candidates across all
  historical scenarios instead of three.
- Raise the diversity trigger floor and mutation boost so collapse is detected
  earlier.
- Inject a small fraction of warm-start “immigrant” policies into each new
  generation. These policies are mutated from the pretrained policy rather
  than copied from the current elite pool.
- Increase draft exploration modestly so policies experience more varied draft
  rooms while remaining mostly exploitative.
- Record the blended selection score and opponent-relative score in every
  generation report.
- Run the expensive all-season candidate audit only on the final vacation
  segment; intermediate segments still write full checkpoints and generation
  reports.

## Alternatives Considered

### Select by raw full-season fitness only

Rejected because a single favorable historical scenario can dominate selection
and does not measure consistency against normal opponents.

### Increase population size only

Rejected as the first response because it increases runtime without directly
addressing population collapse or scenario difficulty normalization.

### Use fully random drafts

Rejected because uncontrolled randomness would obscure whether the policy is
improving. Exploration remains bounded to the top-ranked draft candidates.

## Consequences

- New runs expose `baseline_relative_weight`, `immigrant_fraction`, and the
  expanded final-candidate audit in their configuration and logs.
- Existing checkpoints remain loadable through backward-compatible metadata
  defaults; resumed runs use the settings recorded in their checkpoint.
- The next overnight run will cost somewhat more in final evaluation, but the
  quality comparison is more trustworthy. Intermediate segmented runs avoid
  repeating that audit, offsetting the larger final candidate set.
- A higher diversity score is not itself proof of a better fantasy manager;
  chronological holdout evaluation remains required.
