# Project documentation

This directory contains the project’s durable explanations and decisions. Generated run output belongs in `reports/` and `logs/`; source code belongs in the top-level packages.

## Start here

- [Architecture](ARCHITECTURE.md): how data, models, agents, the league engine, and reports connect.
- [Spec-Driven Development](SPEC_DRIVEN_DEVELOPMENT.md): how requirements become generated code, tests, experiments, and documentation.
- [Data and reproducibility](DATA_AND_REPRODUCIBILITY.md): season cutoffs, leakage rules, data artifacts, and experiment reporting.
- [Roadmap](ROADMAP.md): current capabilities and next milestones.
- [Decision records](decisions/README.md): why major design choices were made.

## Documentation principles

- State what is implemented, experimental, planned, or unavailable.
- Explain why a design exists, not just what a file is called.
- Keep commands close to the workflow they operate.
- Never put credentials, private league data, or machine-local secrets in documentation.
