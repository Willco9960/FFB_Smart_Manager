# ADR-002: Assistant coach instead of unattended transaction bot

## Status

Accepted

## Context

Fantasy platforms can change authentication and transaction behavior. An automated action can also make an expensive roster mistake. The useful product goal is decision support with clear explanations and human control.

## Decision

The numerical engine may recommend drafts, lineups, waivers, and trades. The league engine validates every action. Any future live-platform integration begins read-only and requires explicit human approval before a real transaction.

## Consequences

- The simulator can still train and compare policies without platform credentials.
- UI work should emphasize evidence, projections, uncertainty, and an approval preview.
- The language model may explain structured output but cannot override rules or submit transactions.
