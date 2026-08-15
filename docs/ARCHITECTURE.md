# Architecture

## System overview

```text
Data loaders
    -> processed seasons and pre-decision feature rows
    -> projection models
    -> manager policies
    -> league engine
    -> weekly results, transactions, standings, playoffs
    -> replay and fitness reports
    -> evolution and walk-forward evaluation
```

## Module responsibilities

### `fantasy_engine/`

Owns the simulated world. It defines players, teams, leagues, scoring, drafts, lineups, seasons, playoffs, weekly schedules, waivers, trades, and ownership state. This layer is the authority for legal actions and points.

### `models/` and `llm/`

Contain prediction components and the optional local explanation model. Numerical models produce structured scores or action features. The local language model is an explainer, not a replacement for the rules engine.

### `agents/`

Contain policies that select actions from the current state. Agents can be random, projection-driven, genome-driven, or neural. They must ask the league engine to validate actions.

### `evolution/`

Contains population evaluation, self-play, behavior cloning, transaction replay, value-model validation, mutation, crossover, and walk-forward evaluation. It should compare baselines and ablations rather than assuming a neural or hybrid policy is better.

### `scripts/`

Are executable research entry points. A script should state its inputs, output paths, seed, seasons, and major simplifications in its terminal report.

### `ui/`

Is currently an offline prototype. It demonstrates navigation and presentation using placeholder data; it is not yet the ESPN-connected desktop product.

## Decision flow

At each simulated decision point:

1. Load the historical state available before the decision.
2. Build features and projections.
3. Ask the policy for a candidate action.
4. Validate the action using league rules.
5. Record the action, rejection reason if applicable, and pre-decision features.
6. Advance the simulated calendar.
7. Apply actual historical outcomes only after the decision.
8. Attribute downstream value to the original action in replay reports.

## Training flow

```text
Historical seasons
       |
       +--> supervised projection checkpoints
       |
       +--> season simulator and replay records
                         |
                         v
                 policy population
                         |
                         v
             fitness, baselines, ablations
                         |
                         v
                 select / cross / mutate
                         |
                         v
                 next generation
```

Historical scenario evaluation is process-parallel because the dominant work is Python draft, lineup, waiver, trade, and matchup simulation rather than large neural tensor batches. Persistent workers cache the historical scenario library once and receive only policy payloads plus scenario indices on later generations, avoiding repeated season-data serialization. Each worker returns results aligned to the original agent order, and `--evaluation-workers` bounds CPU and memory usage. Generation telemetry reports Petic GPH (generations per hour).

The opt-in island trainer adds a second parallelism strategy. Independent
populations run bounded generation segments in separate processes. A segment
barrier exchanges elite policies in a ring before the next segment begins. This
preserves generation dependencies inside each island while allowing multiple
search trajectories to progress concurrently. Island workers deliberately run
one scenario worker each; nesting both process pools would oversubscribe the
machine.

All modular policies must be checked with a chronological holdout evaluator
before selection. The holdout path uses the same leakage-safe season loader and
reports weekly wins, points for, playoff rate, championship rate, and
transaction reward.

Pretraining follows a hybrid device plan: CUDA handles contiguous tensor
batches, while the Python-heavy simulator remains on bounded CPU workers. The
trainer reuses those workers across generations and limits each worker's
PyTorch thread pools to avoid multiplying BLAS threads across processes.

The `gpu_sim/` package is an isolated tensorized prototype. It currently
benchmarks projection-best snake drafts and offensive lineup scoring in large
batches, with a CPU reference and CUDA parity tests. It is not wired into the
full-season simulator: waivers, trades, playoffs, and irregular league rules
remain on the production CPU path until each component passes an equivalent
parity and outcome-validation gate. Use
`scripts.compare_tensorized_backends` to measure the prototype and do not
interpret its batch rate as full-season Petic GPH.

The prototype's migration boundary is a `TensorScenarioBatch`: player
projections, actual outcomes, position IDs, and stable player keys are loaded
once and moved together to the target device. Draft kernels reuse an in-place
score buffer, and `--profile-stages` enables synchronized draft/lineup timing
without adding overhead to ordinary throughput runs. The first profile showed
lineup selection as the larger stage, so it is the next optimization target.

The full-stage prototype in `gpu_sim/full_season.py` covers tensorized draft,
weekly scoring, inverse-standings waivers, one-for-one trades, standings, and
the six-team playoff bracket. Standings include wins, losses, ties,
points-for, and points-against. It is benchmark-only until parity reports are
accepted. Its waiver and trade counters do not yet replace the CPU replay
reward attribution. Early measurements show that CUDA needs a large batch: 8 leagues can
be slower than CPU, while 256 leagues are faster. The intended production
batch therefore combines manager population members with historical scenarios
instead of dispatching one league at a time.

`gpu_sim/historical_adapter.py` converts leakage-safe previous-season
projections and week-by-week outcomes into the same tensor contract. Use
`scripts/run_cuda_cpu_historical_comparison.py` for a multi-season gate: its
default mode requires exact standings and champion parity, while
`--transactions` produces a labeled outcome-delta report until waiver/trade
action parity is complete.

## Important boundaries

- A projection model does not automatically become a manager policy.
- A strong full-season score does not prove weekly consistency or future generalization.
- A neural transaction proposal cannot bypass a genome/rules fallback or validation gate.
- A report must identify when K/DST or transaction behavior is simplified.
