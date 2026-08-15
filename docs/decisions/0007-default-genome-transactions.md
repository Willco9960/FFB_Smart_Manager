# ADR-007: Default vacation runs to genome transactions

- Status: Accepted
- Date: 2026-08-13

## Context

The completed 100-generation vacation experiment compared neural, genome, hybrid, and disabled transaction arms. The genome arm repeatedly produced the strongest risk-adjusted transaction results, while neural and hybrid arms added runtime without demonstrating better generalization.

## Decision

New segmented vacation runs default to `--transaction-mode genome`. Neural and hybrid modes, replay collection, and transaction ablations remain available as explicit command-line options for controlled comparison runs.

## Consequences

- Default runs spend compute on the empirically stronger transaction behavior.
- Neural transaction research is still possible, but it must be intentional and measured.
- The final report continues to record the selected transaction mode and ablation metrics when requested.
