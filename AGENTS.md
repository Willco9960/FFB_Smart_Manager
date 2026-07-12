# Fantasy Football AI Manager — Project Instructions

## Project goal

Build a Windows fantasy-football assistant that gathers league and player information, simulates complete historical fantasy seasons, and gives the user explainable recommendations for drafts, lineups, waivers, and trades.

The finished product is an assistant coach. It should recommend actions and require human approval before changing a real ESPN team. Do not design it as an unattended ESPN transaction bot.

## Simulator end state

The historical simulator must model the full season lifecycle:

1. Draft a 10-team league.
2. Set legal weekly lineups.
3. Process waiver claims.
4. Process realistic trades.
5. Score weekly head-to-head matchups.
6. Track standings and points for.
7. Run the playoff bracket and championship.
8. Record decisions, outcomes, and fitness metrics.
9. Use complete-season results to evaluate and evolve manager policies.

Manager agents should be different from one another. The evolutionary system should select strong policies, crossbreed them, mutate them, and test them again across historical seasons.

## Data leakage rules

Historical decisions must only use information available before the decision:

- A 2021 draft may use 2020 results and preseason projections, but never 2021 results.
- Week N lineup, waiver, and trade decisions may use weeks before N, but never week N or future weeks.
- Actual scores are applied only after the decision is made.
- Walk-forward evaluation is preferred over random train/test splits.

If a feature would not have been known by a real manager at that time, do not use it.

## Fantasy rules

Use the project’s ESPN-style defaults unless a league-specific setting overrides them:

- 10 teams
- 14 regular-season weeks
- 6 playoff teams with 2 first-round byes
- QB, 2 RB, 2 WR, TE, FLEX, K, and DST starting slots
- 16-player rosters

The current historical simulator may temporarily use offensive-only slots when K/DST data is unavailable, but this limitation must be clearly reported and not silently treated as ESPN-complete.

Transactions must preserve legal ownership and roster state. Every draft pick, waiver move, trade, lineup decision, rejection, and result should be traceable in reports.

## Architecture principles

Keep responsibilities separate:

- Projection model: predicts player value.
- Manager policy: chooses draft, lineup, waiver, or trade actions.
- League engine: enforces roster, scoring, schedule, and transaction rules.
- Evolution engine: evaluates policies and creates new generations.
- Reports/UI: explains decisions and outcomes.

Do not use a neural network where a deterministic rules or optimization component is more appropriate. Lineup legality and transaction validation belong in the league engine.

## Development rules

- Use Python 3.12+ and the project virtual environment.
- Use Ruff for linting and formatting.
- Use pytest for every new behavior.
- Run Ruff and the full test suite before committing.
- Preserve existing user changes and do not use destructive Git commands.
- Prefer small, testable modules over large scripts.
- Keep generated datasets, model weights, caches, and secrets out of Git.
- Never commit ESPN credentials, cookies, `SWID`, `espn_s2`, or `.env` files.
- Use clear terminal reports so the user can inspect decisions week by week.

## Current priorities

1. Finish and validate the complete historical simulator.
2. Improve waiver and trade realism, including deadlines, limits, and multi-player trades.
3. Add historical K/DST data and full ESPN lineup scoring.
4. Add decision and fitness reports for evolutionary training.
5. Connect the recommendation engine to read-only ESPN synchronization.
6. Build the dashboard and human-approval workflow.

## Communication preferences

Explain the result first, then the implementation details. Be honest about what is simulated, what is trained, and what remains simplified. When teaching, explain why a design choice matters instead of only giving commands.
