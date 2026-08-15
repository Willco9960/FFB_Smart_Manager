# ADR-006: Preserve interrupted segment reports and selected transaction mode

- Status: Accepted
- Date: 2026-08-13

## Context

The vacation run completed successfully, but interruptions exposed two observability problems. A retry could overwrite the generation records written before an interruption, and the final report could print the requested hybrid score while selecting the genome transaction arm after ablation.

## Decision

Resume attempts load and extend an existing segment report instead of replacing it. Final evaluation reports explicitly record the selected transaction mode and both the selected arm score and the underlying full-evaluation score.

## Consequences

- Interrupted segments retain all generation evidence across retries.
- Reports distinguish training fitness, full-evaluation fitness, and the transaction arm actually recommended for deployment.
- Existing report files remain compatible because missing fields are added only on the next resume/evaluation.
