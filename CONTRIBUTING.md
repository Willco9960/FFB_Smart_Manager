# Contributing

Thank you for improving the Fantasy Football AI Manager. Contributions should make the simulator more correct, more reproducible, or easier to understand.

## Spec-Driven Development

This is a 100% Spec-Driven Development project. The user describes the behavior and acceptance criteria; coding agents implement the Python, neural-network, simulator, and documentation changes. Contributors should write the specification and validation plan clearly enough that an agent can implement it and another person can verify it.

The user is not expected to hand-program the project. Explanations, tests, reports, and honest limitations are part of the deliverable—not optional polish after the code.

## Before you start

Read:

- [README.md](README.md) for the public project overview.
- [AGENTS.md](AGENTS.md) for code and simulation invariants.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module ownership.
- [docs/DATA_AND_REPRODUCIBILITY.md](docs/DATA_AND_REPRODUCIBILITY.md) before changing historical features.

## Development setup

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install duckdb torch pytest ruff
```

## Change workflow

1. Create a focused branch.
2. Describe the behavior or bug you are changing.
3. Add or update tests first when practical.
4. Keep data transformations and decision boundaries explicit.
5. Run formatting, linting, and the full test suite.
6. Document new assumptions or architectural choices.
7. Report real, synthetic, and holdout results separately.

## Required checks

```powershell
ruff format .
ruff check .
pytest
```

For model or simulator changes, include the command, seed, seasons, population, generations, runtime, baseline, and holdout result in the pull request or experiment report.

For long-running work, state whether a job can resume after interruption. If it cannot restore the full training state, describe it as a new independent run rather than a continuation.

## Data and secrets

Do not commit downloaded datasets, model checkpoints, logs, `.env` files, ESPN cookies, or API keys. Use `.env.example` for non-secret configuration names only. If a test needs data, use a small deterministic fixture under `tests/`.

## Pull requests

Explain:

- What changed and why.
- Which invariants are preserved.
- How the change was tested.
- Any known limitations or follow-up work.
- Whether results improved against a baseline or only demonstrate that the pipeline runs.

Avoid claims such as “wins every league” unless they are supported by a clearly described, chronological holdout experiment.
