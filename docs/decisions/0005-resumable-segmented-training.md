# ADR-005: Resumable segmented training runs

- Status: Accepted
- Date: 2026-07-27

## Context

The modular self-play trainer can run for several hours. A single long process is fragile: a terminal closure, update, power interruption, or transient error can waste completed generations. The project owner also wants to use a multi-day vacation window without treating an uninterrupted process as a requirement.

## Decision

Save an atomic full training-state checkpoint after every completed generation and expose a resume CLI. The checkpoint includes the population, best policy and candidates, random-number-generator state, generation cursor, elapsed time, and run metadata. Use `scripts/run_modular_vacation_training` to divide long runs into bounded segments, write a manifest, preserve reports/log locations, and invoke the resume CLI for later segments.

## Consequences

- An interrupted run can continue from the last completed generation.
- A generation interrupted halfway is replayed, which is safer than pretending partial work is valid.
- Long runs have durable state and human-readable progress, but still need chronological holdout evaluation afterward.
- More training is not assumed to improve the policy; baselines, diversity, risk metrics, and holdout results remain required.
