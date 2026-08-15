# Spec-Driven Development

## Project policy

The Fantasy Football AI Manager is built using **100% Spec-Driven Development (SDD)**. The user does not hand-write the Python, PyTorch, simulator, or UI implementation. Instead, the user defines the desired behavior and constraints, and coding agents generate and modify the implementation.

This is not a claim that generated code is automatically correct. It changes where correctness work happens: requirements, acceptance criteria, tests, reports, review, and documentation must make every change inspectable.

## Roles

### User

The user owns:

- Product goals and priorities.
- Fantasy-league rules and preferences.
- Safety boundaries for real-platform actions.
- Acceptance criteria for a feature.
- Approval of architectural trade-offs and risky experiments.
- Review of terminal output, reports, and explanations.

### Coding agent

The agent owns:

- Inspecting the existing repository before changing it.
- Translating the specification into small implementation steps.
- Writing code, tests, reports, and documentation.
- Preserving data-leakage, legality, and security invariants.
- Running verification commands and reporting failures honestly.
- Updating the public documentation whenever behavior or assumptions change.

### Repository

The repository is the shared memory of the project. A completed feature is not complete until the code, tests, experiment instructions, limitations, and decision rationale agree.

## Required change loop

```text
1. Specification
   Define behavior, inputs, outputs, constraints, and acceptance tests.

2. Repository inspection
   Locate existing modules, conventions, tests, reports, and known limitations.

3. Design
   Choose the smallest compatible change; write an ADR when the choice is costly to reverse.

4. Implementation
   Generate the code in small modules without bypassing deterministic rule checks.

5. Verification
   Run tests, linting, targeted experiments, and leakage/legality checks.

6. Evidence
   Save commands, seeds, runtime, metrics, checkpoints, and holdout results.

7. Documentation
   Update README, contributor guidance, architecture docs, roadmap, changelog, and ADRs as applicable.
```

## Acceptance criteria template

Every substantial request should answer:

```text
Goal: What user-visible capability should exist?
Inputs: What data and configuration are allowed?
Outputs: What files, reports, UI state, or recommendations should appear?
Rules: What must never happen?
Baseline: What existing behavior is being compared?
Validation: Which tests and holdout experiment demonstrate success?
Runtime: What time and hardware budget is acceptable?
Fallback: What happens if data or a model validation gate fails?
```

## Model-training requirements

Generated training code must identify:

- The information cutoff for every feature.
- Training, validation, and holdout seasons.
- Random seed and population configuration.
- Baselines and ablations.
- Fitness definition and known sources of luck.
- Checkpoint contents and whether they are actually resumable.
- A safe fallback when a neural component fails validation.

## User learning and explanations

The user is building this project to learn through an AI-assisted workflow. Explanations should therefore state:

- What the agent changed.
- Why the change was necessary.
- What remains simplified or uncertain.
- How the user can run and inspect it.
- Which result is evidence versus an unverified hypothesis.

The agent must not pretend that a generated model is skilled merely because a training loss decreased or one simulated season was won.

## Public transparency

SDD does not authorize publishing secrets. Keep credentials and private league data out of the repository while publishing the architecture, assumptions, commands, limitations, and non-sensitive experiment evidence.
