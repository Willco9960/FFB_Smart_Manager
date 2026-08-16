"""CUDA evolutionary training for a manager policy.

This is the first training path that uses the CUDA season simulator for the
fitness loop.  A candidate policy controls one rotating team per scenario;
the other nine teams are projection-best baselines.  Waiver and trade stages
remain enabled in the tensorized season engine.
"""

from __future__ import annotations

import copy
import importlib.util
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from gpu_sim.full_season import CudaSeasonState, run_full_cuda_season
from models.modular_manager_policy import ModularManagerPolicyNetwork

WEEKLY_WIN_REWARD = 15.0
POINTS_FOR_WEIGHT = 0.05
PLAYOFF_QUALIFICATION_REWARD = 40.0
PLAYOFF_WIN_REWARD = 30.0
CHAMPIONSHIP_REWARD = 150.0


@dataclass(frozen=True)
class CudaPolicyEvaluation:
    fitness: float
    fitness_stddev: float
    risk_adjusted_fitness: float
    wins: float
    points_for: float
    playoff_rate: float
    championship_rate: float
    elapsed_seconds: float


@dataclass(frozen=True)
class CudaGenerationMetrics:
    generation: int
    generations: int
    average_fitness: float
    best_fitness: float
    best_fitness_stddev: float
    best_risk_adjusted_fitness: float
    best_wins: float
    best_points_for: float
    best_playoff_rate: float
    best_championship_rate: float
    elapsed_seconds: float
    generations_per_hour: float

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def clone_cuda_state(
    state: CudaSeasonState,
    *,
    scenario_repeats: int = 1,
    projection_noise: float = 0.015,
    seed: int = 1,
) -> CudaSeasonState:
    """Repeat one historical season into randomized, leakage-safe scenarios."""

    if scenario_repeats < 1:
        raise ValueError("scenario_repeats must be positive.")
    if projection_noise < 0.0:
        raise ValueError("projection_noise cannot be negative.")
    device = state.device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draft = state.draft_projections.repeat(scenario_repeats, 1)
    weekly_projection = state.weekly_projections.repeat(scenario_repeats, 1, 1)
    actual = state.weekly_actual_points.repeat(scenario_repeats, 1, 1)
    if projection_noise:
        draft_noise = torch.randn(draft.shape, generator=generator) * projection_noise
        weekly_noise = torch.randn(weekly_projection.shape, generator=generator) * projection_noise
        draft = draft * (1.0 + draft_noise.to(device)).clamp_min(0.50)
        weekly_projection = weekly_projection * (1.0 + weekly_noise.to(device)).clamp_min(0.50)
    return CudaSeasonState(
        draft_projections=draft,
        weekly_projections=weekly_projection,
        weekly_actual_points=actual,
        positions=state.positions,
        team_count=state.team_count,
        roster_size=state.roster_size,
    )


def _candidate_fitness(
    state: CudaSeasonState,
    candidate_team_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = torch.arange(state.scenario_count, device=state.device)
    wins = state.wins[batch, candidate_team_indices].to(torch.float32)
    points_for = state.points_for[batch, candidate_team_indices]
    ranking_value = state.wins.to(torch.float32) * 100000.0 + state.points_for
    seeds = torch.argsort(ranking_value, dim=1, descending=True)
    playoff_qualified = (seeds[:, :6] == candidate_team_indices.unsqueeze(1)).any(dim=1)
    playoff_wins = state.playoff_wins[batch, candidate_team_indices].to(torch.float32)
    champion = (state.champions == candidate_team_indices).to(torch.float32)
    fitness = (
        wins * WEEKLY_WIN_REWARD
        + points_for * POINTS_FOR_WEIGHT
        + playoff_qualified.to(torch.float32) * PLAYOFF_QUALIFICATION_REWARD
        + playoff_wins * PLAYOFF_WIN_REWARD
        + champion * CHAMPIONSHIP_REWARD
    )
    return fitness, wins, points_for, playoff_qualified, champion


@torch.inference_mode()
def evaluate_cuda_policy(
    policy: ModularManagerPolicyNetwork,
    historical_states: list[CudaSeasonState],
    *,
    scenario_repeats: int = 8,
    projection_noise: float = 0.015,
    enable_transactions: bool = True,
    seed: int = 1,
    draft_anchor_weight: float = 0.20,
    risk_penalty: float = 0.10,
    compile_policy: bool = False,
) -> CudaPolicyEvaluation:
    """Evaluate one policy across historical seasons on the selected device."""

    if not historical_states:
        raise ValueError("At least one historical state is required.")
    if risk_penalty < 0.0:
        raise ValueError("risk_penalty cannot be negative.")
    policy.eval()
    device = historical_states[0].device
    policy.to(device)
    scoring_policy = policy
    if compile_policy and device.type == "cuda":
        if importlib.util.find_spec("triton") is None:
            warnings.warn(
                "torch.compile requested but Triton is unavailable; "
                "using eager CUDA policy forwards.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            try:
                scoring_policy = torch.compile(policy, mode="reduce-overhead", dynamic=False)
            except Exception as error:  # pragma: no cover - depends on local Triton install
                warnings.warn(
                    f"torch.compile unavailable; using eager CUDA policy forwards: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    started = perf_counter()
    fitness_values = []
    wins_values = []
    points_values = []
    playoff_values = []
    champion_values = []
    for season_index, template in enumerate(historical_states):
        state = clone_cuda_state(
            template,
            scenario_repeats=scenario_repeats,
            projection_noise=projection_noise,
            seed=seed + season_index,
        )
        candidate_team_indices = (
            torch.arange(state.scenario_count, device=device) % state.team_count
        )
        run_full_cuda_season(
            state,
            enable_transactions=enable_transactions,
            policy_network=scoring_policy,
            policy_team_indices=candidate_team_indices,
            draft_anchor_weight=draft_anchor_weight,
        )
        fitness, wins, points_for, playoff, champion = _candidate_fitness(
            state, candidate_team_indices
        )
        fitness_values.append(fitness.mean())
        wins_values.append(wins.mean())
        points_values.append(points_for.mean())
        playoff_values.append(playoff.to(torch.float32).mean())
        champion_values.append(champion.mean())
    elapsed = perf_counter() - started
    fitness_tensor = torch.stack(fitness_values)
    fitness = float(fitness_tensor.mean().item())
    fitness_stddev = float(fitness_tensor.std(unbiased=False).item())
    return CudaPolicyEvaluation(
        fitness=fitness,
        fitness_stddev=fitness_stddev,
        risk_adjusted_fitness=fitness - (risk_penalty * fitness_stddev),
        wins=float(torch.stack(wins_values).mean().item()),
        points_for=float(torch.stack(points_values).mean().item()),
        playoff_rate=float(torch.stack(playoff_values).mean().item()),
        championship_rate=float(torch.stack(champion_values).mean().item()),
        elapsed_seconds=elapsed,
    )


def clone_policy(policy: ModularManagerPolicyNetwork) -> ModularManagerPolicyNetwork:
    return copy.deepcopy(policy)


def mutate_policy(
    policy: ModularManagerPolicyNetwork,
    rng: random.Random,
    strength: float,
) -> ModularManagerPolicyNetwork:
    child = clone_policy(policy)
    with torch.no_grad():
        for parameter in child.parameters():
            scale = parameter.detach().float().std(unbiased=False).item() or 0.01
            noise = torch.randn_like(parameter) * (strength * scale)
            parameter.add_(noise)
    return child


def crossover_policy(
    first: ModularManagerPolicyNetwork,
    second: ModularManagerPolicyNetwork,
    rng: random.Random,
) -> ModularManagerPolicyNetwork:
    child = clone_policy(first)
    with torch.no_grad():
        second_state = second.state_dict()
        child_state = {}
        for name, first_value in first.state_dict().items():
            mask = torch.rand(first_value.shape, device=first_value.device) < 0.5
            child_state[name] = torch.where(mask, first_value, second_state[name])
        child.load_state_dict(child_state)
    return child


def train_cuda_policy_population(
    initial_policy: ModularManagerPolicyNetwork,
    historical_states: list[CudaSeasonState],
    *,
    population_size: int = 16,
    generations: int = 10,
    selection_count: int = 4,
    mutation_strength: float = 0.02,
    final_mutation_strength: float = 0.005,
    scenario_repeats: int = 8,
    projection_noise: float = 0.015,
    enable_transactions: bool = True,
    seed: int = 1,
    draft_anchor_weight: float = 0.20,
    risk_penalty: float = 0.10,
    compile_policy: bool = False,
    generation_callback=None,
) -> tuple[ModularManagerPolicyNetwork, list[CudaGenerationMetrics]]:
    """Evolve neural manager policies using CUDA full-season fitness."""

    if selection_count < 1 or selection_count > population_size:
        raise ValueError("selection_count must be between one and population_size.")
    rng = random.Random(seed)
    population = [clone_policy(initial_policy)]
    while len(population) < population_size:
        population.append(mutate_policy(initial_policy, rng, mutation_strength))
    best_policy = clone_policy(initial_policy)
    best_risk_adjusted = float("-inf")
    metrics: list[CudaGenerationMetrics] = []
    started = perf_counter()

    for generation in range(1, generations + 1):
        evaluations = [
            evaluate_cuda_policy(
                policy,
                historical_states,
                scenario_repeats=scenario_repeats,
                projection_noise=projection_noise,
                enable_transactions=enable_transactions,
                # Common random numbers: every policy sees identical scenario
                # noise and rotating draft slots, so ranking reflects policy
                # decisions rather than an accidental easier draw.
                seed=seed + generation * 1000,
                draft_anchor_weight=draft_anchor_weight,
                risk_penalty=risk_penalty,
                compile_policy=compile_policy,
            )
            for index, policy in enumerate(population)
        ]
        ranked = sorted(
            zip(evaluations, population, strict=True),
            key=lambda item: item[0].risk_adjusted_fitness,
            reverse=True,
        )
        best_evaluation, generation_best = ranked[0]
        if best_evaluation.risk_adjusted_fitness > best_risk_adjusted:
            best_risk_adjusted = best_evaluation.risk_adjusted_fitness
            best_policy = clone_policy(generation_best)
        elapsed = perf_counter() - started
        generation_metrics = CudaGenerationMetrics(
            generation=generation,
            generations=generations,
            average_fitness=sum(item.fitness for item in evaluations) / len(evaluations),
            best_fitness=best_evaluation.fitness,
            best_fitness_stddev=best_evaluation.fitness_stddev,
            best_risk_adjusted_fitness=best_evaluation.risk_adjusted_fitness,
            best_wins=best_evaluation.wins,
            best_points_for=best_evaluation.points_for,
            best_playoff_rate=best_evaluation.playoff_rate,
            best_championship_rate=best_evaluation.championship_rate,
            elapsed_seconds=elapsed,
            generations_per_hour=generation / max(elapsed / 3600.0, 1e-9),
        )
        metrics.append(generation_metrics)
        if generation_callback is not None:
            generation_callback(generation_metrics, best_policy)

        selected = [policy for _, policy in ranked[:selection_count]]
        next_strength = mutation_strength - (
            (mutation_strength - final_mutation_strength)
            * ((generation - 1) / max(generations - 1, 1))
        )
        population = [clone_policy(best_policy)]
        while len(population) < population_size:
            first = rng.choice(selected)
            second = rng.choice(selected)
            child = crossover_policy(first, second, rng)
            population.append(mutate_policy(child, rng, next_strength))

    return best_policy, metrics


def save_cuda_policy_checkpoint(
    policy: ModularManagerPolicyNetwork,
    output_path: Path,
    metrics: list[CudaGenerationMetrics],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "player_feature_count": policy.player_feature_count,
            "state_feature_count": policy.state_feature_count,
            "hidden_size": policy.hidden_size,
            "state_dict": {
                name: value.detach().cpu() for name, value in policy.state_dict().items()
            },
            "metrics": [item.to_dict() for item in metrics],
        },
        output_path,
    )
    return output_path
