# ADR-001: Public project boundaries

## Status

Accepted

## Context

The repository is intended to be an open research project. Readers need to understand what is implemented, what is experimental, and what remains simplified. Public documentation must not expose credentials or private league data.

## Decision

Keep source code, non-secret configuration names, architecture, assumptions, training commands, evaluation reports, and known limitations public. Keep credentials, cookies, private league identifiers, large generated datasets, caches, and local secrets out of Git.

## Consequences

- A public reader can reproduce the workflows without receiving private access.
- Documentation must label demo, real-season, synthetic, and holdout results.
- `.env.example` may describe configuration names but never contain real values.
- A future license decision is still required before claiming broad reuse rights.
