# Roadmap

The roadmap is ordered by correctness and evidence, not by visual novelty.

## Vacation-run readiness

The modular trainer is ready for bounded, supervised multi-day experiments. It writes an atomic generation-boundary checkpoint containing the population, candidate policies, RNG state, metadata, elapsed time, and generation cursor. The segmented vacation runner writes a recovery manifest and resumes completed segments when the same run ID is reused. In-flight generation work is intentionally replayed after interruption.

Before using a ten-day vacation window, verify:

1. A small fresh-plus-resume smoke run completes.
2. `Tee-Object` output is visible and the manifest is updated.
3. The holdout season remains excluded from training.
4. A final cross-run selection and chronological holdout evaluation are scheduled after the run.

## Working foundation

- Historical draft and weekly season simulation.
- Lineup optimization and head-to-head scoring.
- Waiver and trade simulation with ownership tracking.
- Policy evolution, replay rewards, and walk-forward evaluation.
- Projection checkpoints and optional local explanation model.
- Offline cockpit-style UI prototype.

## Next priorities

1. Finish reliable historical kicker and defense data so the default lineup is fully ESPN-complete.
2. Expand transaction realism: deadlines, waiver priority/faab variants, trade windows, roster limits, and rejection reasons.
3. Improve transaction ablations and require chronological validation before neural transaction arms influence self-play.
4. Add richer decision reports that show draft order, weekly lineups, waiver drops/adds, trade value, and downstream points.
5. Connect the UI to local simulation reports before adding platform synchronization.
6. Add read-only league connectors, then human-approved action previews.
7. Package the Windows application only after the engine and safety boundaries are stable.
8. Compare multiple vacation-run segments and select the final policy using chronological holdout metrics.

## Explicit non-goals for the current phase

- Blind automatic ESPN transactions.
- Claims that the model will finish first every season.
- Hiding simplified scoring or missing data behind a polished UI.
- Replacing deterministic roster legality with a neural network.
