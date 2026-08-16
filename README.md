# Fantasy Football AI Manager

An open, research-oriented fantasy-football assistant that simulates complete seasons, trains and evaluates decision policies, and explains draft, lineup, waiver, and trade recommendations.

The project is intentionally transparent: the simulator, training loop, data assumptions, evaluation reports, limitations, and security boundaries are documented in this repository. It is not currently a guaranteed winning strategy or an unattended ESPN transaction bot.

## Development model: 100% spec-driven development

This project is built through **Spec-Driven Development (SDD)**. The user directs the goals, constraints, experiments, and acceptance criteria; Codex and other coding agents translate those specifications into code, tests, reports, and documentation. The user is not expected to hand-write the Python or neural-network implementation.

That is an intentional learning and transparency choice, not a hidden detail. Every generated change should be reviewable through its specification, tests, experiment command, report, and documentation update. See [docs/SPEC_DRIVEN_DEVELOPMENT.md](docs/SPEC_DRIVEN_DEVELOPMENT.md).

## Status

| Area | Current state |
| --- | --- |
| Historical season simulation | Working for the supported datasets |
| Snake draft simulation | Working with randomized draft order and agent policies |
| Weekly lineups and head-to-head scoring | Working in the historical simulator |
| Waivers and trades | Implemented as experimental, traceable simulator actions |
| Neural projections | Working checkpoints for draft and weekly projections |
| Evolutionary/self-play policy training | Working experimental pipeline |
| Walk-forward evaluation | Working and preferred for model comparisons |
| ESPN connection | Planned read-only integration; not required for local simulation |
| Desktop UI | Offline prototype with placeholder/demo state |
| K/DST historical coverage | Incomplete in some workflows; reports identify when offensive-only scoring is used |

## What this project is

The system separates five responsibilities:

1. **Projection models** estimate player outcomes using only information available before a decision.
2. **Manager policies** choose actions such as draft picks, lineups, waivers, and trades.
3. **The league engine** enforces roster legality, scoring, schedule, ownership, and transaction rules.
4. **The evolution engine** evaluates different policies, selects strong candidates, and creates new generations through crossover and mutation.
5. **Reports and the UI** make decisions, outcomes, and limitations inspectable.

This separation is deliberate. A neural network predicts or ranks actions; deterministic code remains responsible for rules that must never be violated.

## Quick start

The supported development environment is Windows PowerShell with Python 3.12 or newer.

```powershell
git clone <repository-url>
cd FFB_Manager
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install duckdb torch pytest ruff
```

Run the small local simulator:

```powershell
python main.py
```

Launch the offline UI prototype:

```powershell
python -m scripts.run_ui
```

Run the quality checks:

```powershell
ruff format --check .
ruff check .
pytest
```

Benchmark the isolated CUDA simulation path:

```powershell
python -m scripts.compare_tensorized_backends --scenarios 4096 --players 512 --repeats 3
```

This writes `reports/tensorized_backend_comparison.json` and reports CPU versus
CUDA batch throughput. The prototype must pass parity tests before it can be
expanded to waivers, trades, playoffs, or production training.

Benchmark the experimental full-season CUDA stages:

```powershell
python -u -m scripts.benchmark_full_cuda_suite --device cuda --scenarios 256 --players 256 --duration-seconds 3600 --progress-seconds 30 --output reports/full_cuda_suite_1h.json 2>&1 | Tee-Object -FilePath logs/full_cuda_suite_1h.console.log
```

This includes draft, weekly scoring, waivers, one-for-one trades, standings,
and playoffs. It is a simulator throughput benchmark, not a learning run.

### Run CUDA manager-policy training

The CUDA trainer uses leakage-safe historical seasons, puts one neural policy
against nine projection-best opponents, rotates the candidate's draft slot,
and evolves policy weights with crossover and mutation. CUDA waivers and trades
remain enabled in the season fitness loop. The default path keeps exact
per-policy head evaluation for auditability; `--batched-policy-heads` enables
the parity-tested flattened `torch.func.vmap` population route for throughput.
Both modes rank candidates with common randomized scenarios plus a
risk-adjusted score.

Smoke test:

```powershell
python -u -m scripts.train_cuda_manager_policy --device cuda --start-season 2021 --end-season 2021 --population 4 --generations 2 --selection 2 --scenario-repeats 4 2>&1 | Tee-Object -FilePath logs/cuda_manager_smoke.log
```

Flagship overnight profile:

```powershell
python -u -m scripts.train_cuda_manager_policy --device cuda --batched-policy-heads --start-season 2001 --end-season 2024 --population 10 --generations 150 --selection 4 --scenario-repeats 2 --loader-workers 4 --projection-noise 0.015 --draft-anchor-weight 0.20 --risk-penalty 0.10 --holdout-season 2025 --output data/models/cuda_manager_policy_flagship.pt --checkpoint data/models/cuda_manager_training_state_flagship.pt --report reports/cuda_manager_flagship.json 2>&1 | Tee-Object -FilePath logs/cuda_manager_flagship.log
```

The best-policy checkpoint is written after every generation to the configured
`--output`; the full population/RNG resume checkpoint is written to
`--checkpoint`; progress and the 2025 audit are written to `--report`.

If the process stops, resume from the last completed generation:

```powershell
python -u -m scripts.train_cuda_manager_policy --device cuda --batched-policy-heads --start-season 2001 --end-season 2024 --population 10 --generations 150 --selection 4 --scenario-repeats 2 --loader-workers 4 --projection-noise 0.015 --draft-anchor-weight 0.20 --risk-penalty 0.10 --holdout-season 2025 --resume data/models/cuda_manager_training_state_flagship.pt --output data/models/cuda_manager_policy_flagship.pt --checkpoint data/models/cuda_manager_training_state_flagship.pt --report reports/cuda_manager_flagship.json 2>&1 | Tee-Object -FilePath logs/cuda_manager_flagship_resume.log
```

Do not pass `--compile-policy` on this Windows installation unless a working
Triton runtime has been installed; the trainer detects its absence and safely
falls back to eager CUDA.

Run a real historical CPU/CUDA parity sweep:

```powershell
python -u -m scripts.run_cuda_cpu_historical_comparison --start-season 2021 --end-season 2024 --players 256 --device cuda 2>&1 | Tee-Object -FilePath logs/cpu_cuda_historical_parity.log
```

The default mode requires exact standings and champion matches. Add
`--transactions` only for the labeled waiver/trade outcome-delta experiment;
those tensorized transaction decisions are not yet action-identical to the CPU
agents.

Run the same benchmark continuously for one hour with live progress:

```powershell
python -u -m scripts.benchmark_tensorized_cuda --device cuda --scenarios 4096 --players 512 --teams 10 --rounds 16 --duration-seconds 3600 --progress-seconds 30 --output reports/tensorized_cuda_1h.json 2>&1 | Tee-Object -FilePath logs/tensorized_cuda_1h.console.log
```

## Common workflows

### Inspect historical data

```powershell
python -m scripts.audit_historical_data
python -m scripts.rebuild_2021_processed_season
```

The audit reports missing files and the available season window. Data files are intentionally kept outside Git when they are large or generated.

### Build projection checkpoints

```powershell
python -m scripts.train_draft_projection_nn
python -m scripts.train_weekly_projection_nn
```

These models are supervised projection components. They are not the complete manager and should not be judged by championship results alone.

### Run a small modular-policy experiment

```powershell
python -u -m scripts.train_modular_manager_policy --start-season 2021 --end-season 2021 --population 10 --generations 1 --selection 3 --epochs 1 2>&1 | Tee-Object -FilePath logs/modular_policy_smoke.log
```

### Run walk-forward evaluation

```powershell
python -u -m scripts.evaluate_manager_walk_forward --start-season 2021 --end-season 2023 --minimum-training-seasons 1 --population 10 --generations 1 --selection 3 --initial-policy data/models/modular_manager_policy.pt 2>&1 | Tee-Object -FilePath logs/manager_walk_forward.log
```

Walk-forward evaluation trains on earlier seasons and evaluates on later unseen seasons. It is the project’s primary defense against accidentally training on future information.

### Run a bounded overnight profile

```powershell
python -u -m scripts.train_modular_manager_policy --start-season 2001 --end-season 2024 --selection 8 --epochs 50 --offline-epochs 50 --collect-season-replay --transaction-ablation --transaction-mode hybrid --transaction-value-epochs 100 --overnight-profile 2>&1 | Tee-Object -FilePath logs/training_modular_overnight.log
```

`Tee-Object` keeps progress visible while saving a copy. Checkpoints, models, reports, and logs are separate from source code so a long run can be inspected and recovered manually without changing the implementation.

### Vacation and multi-day training runs

The modular trainer now writes a complete, atomic generation-boundary state checkpoint. The checkpoint restores the policy population, best candidates, replay-selection state, random-number-generator state, elapsed time, metadata, and generation cursor. A process can therefore resume after a terminal closes or a machine restart without replaying completed generations. An interruption during a generation resumes from the preceding completed generation; in-flight work is intentionally not treated as complete.

Observed profile runtimes in this repository:

| Run | Configuration | Recorded runtime |
| --- | --- | ---: |
| v9 | 24 agents, 10 generations, rotating 12-season profile | 6.99 hours |
| v8 | 30 agents, 12 generations, 24-season workload | 14.24 hours |
| v7 | 30 agents, 12 generations, 24-season workload | 14.28 hours |

The runtime varies with scenario count, transaction replay, ablations, validation, and machine load. Use bounded segments so every segment produces a durable checkpoint and a manifest. The runner automatically continues an interrupted run when the same `--run-id` is reused.

```powershell
python -u -m scripts.run_modular_vacation_training --run-id vacation_2026_01 --segments 10 --generations-per-segment 10 --start-season 2001 --end-season 2024 --population 24 --selection 8 --epochs 50 --offline-epochs 50 2>&1 | Tee-Object -FilePath logs/vacation/vacation_2026_01.log
```

The manifest is written to `data/models/vacation_runs/<run-id>/manifest.json`; the full resumable state is `training_state.pt`, generation checkpoints are in the same run directory, and reports are under `reports/vacation/<run-id>/`. If a run stops, rerun the same command and run ID. Use `--force-restart` only when intentionally discarding that run's progress. For a direct manual continuation, use `scripts.resume_modular_manager_policy` with the manifest's state checkpoint.

To inspect a run without touching it:

```powershell
python -u -m scripts.show_modular_run_status --run-id vacation_2026_01
```

This is suitable for a supervised multi-day vacation run, but it is not a guarantee of model improvement. Keep a chronological holdout season untouched and evaluate the final selected checkpoint after the run. More generations can overfit the historical simulator; compare against baselines and preserve the manifest before selecting a winner.

The vacation runner defaults to `--transaction-mode genome` because the completed 100-generation experiment showed genome transactions outperforming neural and hybrid transaction arms. Use `--transaction-mode hybrid --transaction-ablation --collect-season-replay` only when intentionally running a transaction-model comparison.

Historical scenarios are evaluated in parallel by default with eight persistent worker processes. Each worker caches the historical scenario library once, avoiding repeated season-data serialization between generations. This uses CPU cores for the Python-heavy simulator while the GPU remains available for larger neural training batches. Use `--evaluation-workers 1` for debugging or a smaller value if memory pressure appears. Generation logs report Petic GPH (generations per hour) so throughput changes can be measured instead of guessed.

The modular trainer uses `--training-device auto` by default. On this machine it
selects the RTX 3080 for batched behavior-cloning, replay, and transaction-value
training, then moves the compact policy back to CPU before process-based
simulation. Use `--training-device cpu` for a deterministic CPU-only diagnostic.

### Run the pre-training gate before an expensive run

This is the required fast check for a new data range or checkpoint:

```powershell
python -u -m scripts.run_training_preflight --start-season 2021 --end-season 2024 --device auto --output reports/training_preflight_2021_2024.json
```

The gate validates chronological source coverage, the fitness contract, all
four manager heads, finite behavior-cloning loss, and a synthetic season pass.
It exits nonzero when any check fails. To inspect the teacher warm start alone:

```powershell
python -u -m scripts.run_manager_pretraining_gate --season 2021 --device auto --behavior-epochs 2 --offline-epochs 1
```

### Evaluate a modular policy on an unseen season

Keep the holdout season outside the training range. For example, evaluate the
selected 2025 policy checkpoint against the original policy:

```powershell
python -u -m scripts.evaluate_modular_policy_holdout --model data/models/vacation_runs/vacation_2026_02/generation_checkpoints/selected_full_evaluation_generation_083.pt --baseline-model data/models/modular_manager_policy.pt --holdout-season 2025 --output reports/vacation/vacation_2026_02/holdout_2025.json 2>&1 | Tee-Object -FilePath logs/vacation/vacation_2026_02/holdout_2025.log
```

The report compares fitness, weekly wins, points for, playoff rate,
championship rate, and transaction reward. Do not select a new policy from
training fitness alone.

### Train parallel evolutionary islands

Island training is an opt-in alternative to the standard scenario-parallel
trainer. Each island evolves for a bounded segment, then receives one elite
policy from a neighboring island:

```powershell
python -u -m scripts.train_modular_policy_islands --start-season 2001 --end-season 2024 --islands 10 --island-workers 10 --segments 10 --generations-per-segment 10 --population 24 --selection 8 --scenarios-per-generation 8 --full-evaluation-every 5 2>&1 | Tee-Object -FilePath logs/modular_islands_2001_2024.log
```

The island trainer initializes from `data/models/modular_manager_policy.pt` by
default. Pass `--initial-policy` to use a different pretrained checkpoint.

Do not combine ten island workers with scenario workers inside each island;
the island trainer intentionally uses one scenario worker per island to avoid
CPU oversubscription.

### Benchmark trainer choices

Use a small, controlled run before an overnight experiment:

```powershell
python -u -m scripts.benchmark_modular_trainers --start-season 2021 --end-season 2022 --population 4 --generations 2 --selection 2 --evaluation-workers 4 --islands 2 --island-workers 2 2>&1 | Tee-Object -FilePath logs/modular_trainer_benchmark.log
```

Compare both runtime and chronological holdout performance; a faster trainer
is not automatically a better trainer.

### Run the compatibility policy pipeline

The earlier real-season policy path remains available for comparison with the modular pipeline:

```powershell
python -u -m scripts.train_manager_policy_real_seasons --start-season 2001 --end-season 2024 --holdout-season 2025 --population 30 --generations 12 --selection 10 --consistency-penalty 0.25 2>&1 | Tee-Object -FilePath logs/training_compatibility_policy.log
python -u -m scripts.evaluate_real_policy_walk_forward --start-season 2021 --end-season 2025 2>&1 | Tee-Object -FilePath logs/compatibility_walk_forward.log
```

Use this path as a baseline, not as proof that the newer modular policy is better. Keep its results in a separate report when comparing architectures.

### Inspect a historical weekly season

```powershell
python -u -m scripts.run_2021_weekly_season_simulation 2>&1 | Tee-Object -FilePath logs/weekly_2021_simulation.log
```

The report should identify draft order, weekly lineups, transactions, standings, playoff results, and any simplified scoring rules.

## Reading experiment results

Do not compare a single “best fitness” number in isolation. A credible comparison includes:

- Weekly wins and points for.
- Playoff and championship rate across multiple seeds.
- Lineup efficiency and missed-start counts.
- Waiver and trade downstream value.
- Baseline and ablation results.
- Holdout-season performance.
- Runtime, seed, and active rule set.

Generated JSON and PNG reports belong under `reports/`; long terminal transcripts belong under `logs/`; checkpoints belong under `data/models/`.

## Architecture

```text
Historical data + pre-decision features
                |
                v
        Projection models
                |
                v
        Manager policy agents
       /       |        \
    draft    lineup   transactions
       \       |        /
                v
          League engine
     rules, scoring, schedule,
     waivers, trades, playoffs
                |
                v
       Results and explanations
                |
                v
     Evolution / evaluation loop
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module ownership and data flow.

## Repository map

```text
agents/          Draft, lineup, and transaction decision agents
data/            Raw, processed, replay, and generated model artifacts
evolution/       Training, self-play, evaluation, replay, and fitness logic
fantasy_engine/  League rules, scoring, drafts, lineups, transactions, seasons
llm/             Optional local language-model explanation layer
models/          Projection and policy model definitions
scripts/         Runnable training, evaluation, audit, simulation, and UI commands
tests/           Unit and integration tests
ui/              Offline desktop-shell prototype
docs/            Architecture, reproducibility, decisions, roadmap, and policies
logs/            Local run logs; ignored by Git
reports/         Generated charts and evaluation reports
```

## Data and anti-leakage rules

Every historical decision must use only information that a real manager could have had at that time:

- A 2021 draft can use 2020 results and preseason projections, never 2021 outcomes.
- Week N decisions can use weeks before N, never Week N or future weeks.
- Actual scores are applied only after a decision is made.
- Random splits are not sufficient evidence; chronological and walk-forward tests are preferred.
- Synthetic seasons are augmentation experiments, not substitutes for real holdout seasons.

The detailed data contract is in [docs/DATA_AND_REPRODUCIBILITY.md](docs/DATA_AND_REPRODUCIBILITY.md).

## Known limitations

The project is experimental and open about what is not finished:

- Some historical workflows currently use offensive-only lineup rules when reliable kicker and defense data is unavailable.
- ESPN synchronization is not yet a complete production integration.
- The UI is an offline prototype and currently uses placeholder/demo state.
- Neural projections and manager policies are research checkpoints, not proof of future league dominance.
- Transaction recommendations are evaluated in simulation and require human approval for any future real-league integration.
- Results depend on scoring settings, roster rules, data quality, and the exact historical cutoff.

If a run uses a simplified rule set, the report should say so explicitly.

## Transparency and security

- Do not commit `.env`, ESPN cookies, `SWID`, `espn_s2`, API keys, or private league identifiers.
- `.env.example` contains only non-secret local LLM configuration placeholders.
- The optional local coach model explains structured recommendations; it does not override league rules or submit transactions.
- Generated datasets, checkpoints, caches, and logs are reproducible artifacts, not source-of-truth code.
- Security boundaries and responsible credential handling are documented in [SECURITY.md](SECURITY.md).

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Spec-Driven Development](docs/SPEC_DRIVEN_DEVELOPMENT.md)
- [Data and reproducibility](docs/DATA_AND_REPRODUCIBILITY.md)
- [Roadmap](docs/ROADMAP.md)
- [Decision records](docs/decisions/README.md)
- [Development and contribution guide](CONTRIBUTING.md)
- [Public agent/project instructions](AGENTS.md)
- [Changelog](CHANGELOG.md)
- [Security policy](SECURITY.md)

## Contributing

Small, testable improvements are preferred. Before opening a change:

```powershell
ruff format .
ruff check .
pytest
```

Explain the reason for architectural changes, include tests for new behavior, and report whether a result came from real historical data, synthetic data, or a toy fixture. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

No license has been selected yet. Until one is added, the repository is public for inspection, but reuse and redistribution rights should not be assumed. A license decision is tracked in [docs/decisions/0001-public-project-boundaries.md](docs/decisions/0001-public-project-boundaries.md).
