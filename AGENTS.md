# Fantasy Football AI Manager: contributor and agent guide

This file is public project documentation. It contains development conventions and architecture constraints, not credentials or private league information.

## Spec-driven development is mandatory

This project is **100% Spec-Driven Development**. The user supplies the desired behavior, constraints, priorities, and acceptance criteria. Coding agents produce the implementation, tests, reports, and documentation updates. Do not assume the user is hand-writing Python, PyTorch, or simulator code; explain generated changes in plain language and keep them independently verifiable.

Every non-trivial change should connect:

```text
specification -> implementation -> tests -> experiment/report -> documentation
```

If the specification is ambiguous, preserve the safest existing behavior and document the assumption. Do not silently replace the user’s stated objective with a different optimization target.

## Project goal

Build an explainable fantasy-football assistant that can gather league and player information, simulate historical seasons, and recommend drafts, lineups, waivers, and trades. The intended product is an assistant coach: real-world actions require human approval.

## System boundaries

Keep these responsibilities separate:

- **Projection model:** estimates player outcomes from pre-decision information.
- **Manager policy:** chooses among legal actions.
- **League engine:** enforces roster, scoring, schedule, ownership, and transaction rules.
- **Evolution engine:** evaluates policies and creates new candidates.
- **Reports/UI:** explain inputs, decisions, outcomes, and uncertainty.

Do not use a neural network for deterministic legality checks or scoring rules. If a policy proposes an invalid action, the league engine must reject it and record why.

## Simulator contract

The full-season simulator should model:

1. A 10-team snake draft.
2. Legal weekly lineups.
3. Waiver claims and drops.
4. Realistic trade proposals and acceptance rules.
5. Weekly head-to-head matchups.
6. Standings and points for.
7. A six-team playoff bracket with first-round byes.
8. Decision, transaction, and fitness reports.
9. Evolution and evaluation across complete seasons.

The project’s ESPN-style defaults are 10 teams, 14 regular-season weeks, six playoff teams, 16-player rosters, and QB/2 RB/2 WR/TE/FLEX/K/DST starters. If a workflow lacks reliable kicker or defense data, it may use offensive-only rules temporarily, but the limitation must be visible in the report.

## Anti-leakage contract

- A 2021 draft may use 2020 results and preseason projections, never 2021 outcomes.
- Week N decisions may use information from weeks before N, never Week N or future weeks.
- Apply actual scores only after the decision point.
- Prefer chronological and walk-forward validation over random splits.
- Synthetic seasons are augmentation, not evidence that replaces real holdouts.

## Engineering conventions

- Use Python 3.12+ and the repository virtual environment.
- Use Ruff for formatting and linting.
- Use pytest for every new behavior.
- Preserve user changes; never use destructive Git commands.
- Keep generated data, model weights, caches, logs, and secrets out of Git.
- Never commit `.env`, ESPN cookies, `SWID`, `espn_s2`, or API keys.
- Keep long-run output visible with `python -u` and PowerShell `Tee-Object`; put transcripts under `logs/`.
- Add decision records when a choice is expensive to reverse or changes public behavior.

## Verification checklist

Before declaring a run complete or starting another overnight run, reconcile
the authoritative second-brain ledger at
`E:\Codex_Brain\02 - Projects\FFB Manager Overnight Run To-Do.md`. Every item
must be completed and verified; an unchecked item is a hard stop. Items may be
removed only after implementation plus focused/full/runtime evidence is recorded.

Before committing:

```powershell
ruff format .
ruff check .
pytest
```

For model changes, also report:

- Training and holdout seasons.
- Feature cutoff and leakage controls.
- Population, generations, seeds, and runtime.
- Baseline and ablation comparisons.
- Whether K/DST, waivers, trades, and playoffs were active.
- Where checkpoints, reports, and logs were written.

For long training, also report whether the run is resumable. A generation checkpoint alone is not proof that population state, optimizer state, random state, and the generation cursor can be restored. For the modular trainer, verify the full `training_state.pt` round-trip and record the recovery manifest path.

For long modular runs, record `--evaluation-workers`. The simulator is CPU-heavy; use the bounded process-parallel path for real experiments and `--evaluation-workers 1` when debugging deterministic failures.

For vacation runs, prefer `scripts.run_modular_vacation_training` with a unique `--run-id`. Reusing that run ID resumes completed segments; use `--force-restart` only when intentionally discarding the run. The final checkpoint must still be evaluated on a chronological holdout before it is called an improvement.

## Documentation rules

- README: human-facing purpose, quick start, capabilities, limitations, and links.
- `docs/`: architecture, reproducibility, decisions, and roadmap.
- `AGENTS.md`: commands, conventions, and guardrails for contributors and coding agents.
- Changelog: user-visible changes and notable research changes.
- Never hide a simplification behind polished prose; document it where a reader will find it.
