# ADR-004: Use Spec-Driven Development for the entire project

## Status

Accepted

## Context

The project owner is learning Python, machine learning, and fantasy-sports simulation and does not intend to hand-write the implementation. The project also has many interacting rules, long-running experiments, and safety-sensitive platform integration concerns.

## Decision

Use a 100% Spec-Driven Development workflow. Requirements, constraints, acceptance criteria, tests, experiment reports, and documentation are first-class project artifacts. Coding agents generate the implementation, while the user directs goals, reviews evidence, and approves decisions.

## Alternatives considered

### Hand-written implementation only

Rejected for this project because it does not match the owner’s chosen learning workflow or available time.

### Generated code without specifications

Rejected because it makes model goals, leakage boundaries, legality rules, and regressions difficult to inspect.

## Consequences

- Every meaningful feature needs a written acceptance path.
- Documentation updates are part of implementation, not a later cleanup task.
- Generated code still requires tests, linting, targeted experiments, and human review.
- The project can be open about how it was built without exposing secrets.
