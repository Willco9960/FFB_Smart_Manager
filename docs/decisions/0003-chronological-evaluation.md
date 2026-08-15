# ADR-003: Chronological evaluation and leakage prevention

## Status

Accepted

## Context

Random train/test splits can leak future football information into historical decisions. A policy that sees the outcome it is supposed to predict can look strong while failing in a real season.

## Decision

Use season- and week-aware cutoffs. Train on earlier seasons, validate chronologically, and hold out later seasons. At each weekly decision, construct features before applying the actual result. Reports must record the cutoff and any unavailable data.

## Consequences

- Results are slower and noisier than leaked evaluations, but more credible.
- Synthetic data may augment training but cannot replace real chronological holdouts.
- A validation failure should trigger a documented fallback rather than silently selecting the weaker arm.
