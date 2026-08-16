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
from torch.func import functional_call, vmap

from evolution.opponent_archive import OpponentArchive
from fantasy_engine.fitness_contract import ESPN_FITNESS_CONTRACT, FitnessContract
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
    transaction_reward: float = 0.0
    lineup_efficiency: float = 0.0


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
    best_transaction_reward: float = 0.0
    best_lineup_efficiency: float = 0.0

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


def fork_cuda_state(state: CudaSeasonState) -> CudaSeasonState:
    """Create a mutable simulation copy without regenerating scenarios."""

    return CudaSeasonState(
        draft_projections=state.draft_projections.clone(),
        weekly_projections=state.weekly_projections.clone(),
        weekly_actual_points=state.weekly_actual_points,
        positions=state.positions,
        team_count=state.team_count,
        roster_size=state.roster_size,
    )


def prepare_cuda_scenario_bank(
    historical_states: list[CudaSeasonState],
    *,
    scenario_repeats: int = 8,
    projection_noise: float = 0.015,
    seed: int = 1,
) -> list[CudaSeasonState]:
    """Materialize common-random-number scenarios once for a training run.

    Scenario tensors are immutable inputs; each policy receives a cheap mutable
    fork. This avoids repeating random generation and tensor expansion for every
    candidate while guaranteeing identical scenarios across the population.
    """

    return [
        clone_cuda_state(
            state,
            scenario_repeats=scenario_repeats,
            projection_noise=projection_noise,
            seed=seed + season_index,
        )
        for season_index, state in enumerate(historical_states)
    ]


def _candidate_fitness(
    state: CudaSeasonState,
    candidate_team_indices: torch.Tensor,
    contract: FitnessContract = ESPN_FITNESS_CONTRACT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = torch.arange(state.scenario_count, device=state.device)
    wins = state.wins[batch, candidate_team_indices].to(torch.float32)
    points_for = state.points_for[batch, candidate_team_indices]
    ranking_value = state.wins.to(torch.float32) * 100000.0 + state.points_for
    seeds = torch.argsort(ranking_value, dim=1, descending=True)
    playoff_qualified = (seeds[:, :6] == candidate_team_indices.unsqueeze(1)).any(dim=1)
    playoff_wins = state.playoff_wins[batch, candidate_team_indices].to(torch.float32)
    champion = (state.champions == candidate_team_indices).to(torch.float32)
    transaction_reward = torch.zeros_like(points_for)
    for gains in (*state.waiver_policy_gains, *state.trade_policy_gains):
        transaction_reward = transaction_reward + gains
    lineup_efficiency = torch.zeros_like(points_for)
    for gains in state.lineup_policy_gains:
        lineup_efficiency = lineup_efficiency + gains[
            torch.arange(state.scenario_count, device=state.device),
            candidate_team_indices,
        ]
    fitness = (
        wins * contract.weekly_win_reward
        + points_for * contract.points_for_weight
        + playoff_qualified.to(torch.float32) * contract.playoff_qualification_reward
        + playoff_wins * contract.playoff_win_reward
        + champion * contract.championship_reward
        + transaction_reward * contract.transaction_reward_weight
        + lineup_efficiency * contract.lineup_efficiency_weight
    )
    return fitness, wins, points_for, playoff_qualified, champion


def _candidate_auxiliary_metrics(
    state: CudaSeasonState,
    candidate_team_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = torch.arange(state.scenario_count, device=state.device)
    transaction_reward = torch.zeros(state.scenario_count, device=state.device)
    for gains in (*state.waiver_policy_gains, *state.trade_policy_gains):
        transaction_reward = transaction_reward + gains
    lineup_efficiency = torch.zeros(state.scenario_count, device=state.device)
    for gains in state.lineup_policy_gains:
        lineup_efficiency = lineup_efficiency + gains[batch, candidate_team_indices]
    return transaction_reward, lineup_efficiency


class CudaPolicyEnsemble(torch.nn.Module):
    """Functional, parameter-batched view of a policy population."""

    def __init__(self, policies: list[ModularManagerPolicyNetwork]):
        if not policies:
            raise ValueError("At least one policy is required.")
        super().__init__()
        self.population_size = len(policies)
        self.template = copy.deepcopy(policies[0]).eval()
        first_parameters = dict(policies[0].named_parameters())
        self.stacked_parameters = {
            name: torch.stack(
                [dict(policy.named_parameters())[name] for policy in policies],
                dim=0,
            )
            for name in first_parameters
        }
        self.stacked_buffers = {
            name: torch.stack(
                [dict(policy.named_buffers())[name] for policy in policies],
                dim=0,
            )
            for name in dict(policies[0].named_buffers())
        }

    def forward(
        self,
        player_features: torch.Tensor,
        state_features: torch.Tensor,
        decision_type: str = "draft",
    ) -> torch.Tensor:
        if player_features.shape[0] % self.population_size != 0:
            raise ValueError("Batched policy rows must divide evenly by population size.")
        rows_per_policy = player_features.shape[0] // self.population_size
        player_batches = player_features.reshape(self.population_size, rows_per_policy, -1)
        state_batches = state_features.reshape(self.population_size, rows_per_policy, -1)

        def apply_policy(parameters, buffers, player, state):
            return functional_call(
                self.template,
                (parameters, buffers),
                (player, state),
                {"decision_type": decision_type},
            )

        outputs = vmap(apply_policy, in_dims=(0, 0, 0, 0))(
            self.stacked_parameters,
            self.stacked_buffers,
            player_batches,
            state_batches,
        )
        return outputs.reshape(-1)


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
    scenario_bank: list[CudaSeasonState] | None = None,
    fitness_contract: FitnessContract = ESPN_FITNESS_CONTRACT,
    opponent_policies: list[ModularManagerPolicyNetwork] | None = None,
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
    transaction_values = []
    lineup_efficiency_values = []
    templates = historical_states if scenario_bank is None else scenario_bank
    for template in templates:
        state = fork_cuda_state(template)
        candidate_team_indices = torch.zeros(state.scenario_count, dtype=torch.long, device=device)
        team_policy_networks = None
        if opponent_policies:
            archive = [opponent for opponent in opponent_policies if opponent is not policy]
            if not archive:
                archive = [policy]
            team_policy_networks = [policy]
            team_policy_networks.extend(
                archive[index % len(archive)] for index in range(state.team_count - 1)
            )
        run_full_cuda_season(
            state,
            enable_transactions=enable_transactions,
            policy_network=scoring_policy,
            team_policy_networks=team_policy_networks,
            policy_team_indices=candidate_team_indices,
            draft_anchor_weight=draft_anchor_weight,
            fitness_contract=fitness_contract,
        )
        fitness, wins, points_for, playoff, champion = _candidate_fitness(
            state, candidate_team_indices, fitness_contract
        )
        transaction_reward, lineup_efficiency = _candidate_auxiliary_metrics(
            state, candidate_team_indices
        )
        fitness_values.append(fitness.mean())
        wins_values.append(wins.mean())
        points_values.append(points_for.mean())
        playoff_values.append(playoff.to(torch.float32).mean())
        champion_values.append(champion.mean())
        transaction_values.append(transaction_reward.mean())
        lineup_efficiency_values.append(lineup_efficiency.mean())
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
        transaction_reward=float(torch.stack(transaction_values).mean().item()),
        lineup_efficiency=float(torch.stack(lineup_efficiency_values).mean().item()),
    )


@torch.inference_mode()
def evaluate_cuda_policy_population(
    policies: list[ModularManagerPolicyNetwork],
    historical_states: list[CudaSeasonState],
    *,
    scenario_bank: list[CudaSeasonState] | None = None,
    scenario_repeats: int = 8,
    projection_noise: float = 0.015,
    enable_transactions: bool = True,
    seed: int = 1,
    draft_anchor_weight: float = 0.20,
    risk_penalty: float = 0.10,
    fitness_contract: FitnessContract = ESPN_FITNESS_CONTRACT,
    exact_policy_head_parity: bool = True,
) -> list[CudaPolicyEvaluation]:
    """Evaluate all policies in one flattened CUDA scenario batch."""

    if not policies:
        raise ValueError("At least one policy is required.")
    if not historical_states:
        raise ValueError("At least one historical state is required.")

    # In-season heads are stateful by team.  A flattened ensemble can batch
    # draft rows safely, but silently mixes team-conditioned lineup/waiver/
    # trade rows unless a team-aware router is used.  Use the exact evaluator
    # until that router is available; this is slower but prevents a false
    # population ranking and keeps CPU/CUDA behavior auditable.
    if exact_policy_head_parity:
        return [
            evaluate_cuda_policy(
                policy,
                historical_states,
                scenario_bank=scenario_bank,
                scenario_repeats=scenario_repeats,
                projection_noise=projection_noise,
                enable_transactions=enable_transactions,
                seed=seed,
                draft_anchor_weight=draft_anchor_weight,
                risk_penalty=risk_penalty,
                fitness_contract=fitness_contract,
            )
            for policy in policies
        ]
    if scenario_bank is None:
        scenario_bank = prepare_cuda_scenario_bank(
            historical_states,
            scenario_repeats=scenario_repeats,
            projection_noise=projection_noise,
            seed=seed,
        )
    device = scenario_bank[0].device
    ensemble = CudaPolicyEnsemble(policies).to(device)
    population_size = len(policies)
    season_fitness: list[list[torch.Tensor]] = [[] for _ in policies]
    season_wins: list[list[torch.Tensor]] = [[] for _ in policies]
    season_points: list[list[torch.Tensor]] = [[] for _ in policies]
    season_playoffs: list[list[torch.Tensor]] = [[] for _ in policies]
    season_champions: list[list[torch.Tensor]] = [[] for _ in policies]
    season_transactions: list[list[torch.Tensor]] = [[] for _ in policies]
    season_lineup_efficiency: list[list[torch.Tensor]] = [[] for _ in policies]
    started = perf_counter()

    for template in scenario_bank:
        scenarios = template.scenario_count
        state = CudaSeasonState(
            draft_projections=template.draft_projections.repeat(population_size, 1),
            weekly_projections=template.weekly_projections.repeat(population_size, 1, 1),
            weekly_actual_points=template.weekly_actual_points.repeat(population_size, 1, 1),
            positions=template.positions,
            team_count=template.team_count,
            roster_size=template.roster_size,
        )
        candidate_team_indices = (
            torch.arange(scenarios, device=device) % state.team_count
        ).repeat(population_size)
        run_full_cuda_season(
            state,
            enable_transactions=enable_transactions,
            policy_network=ensemble,
            policy_team_indices=candidate_team_indices,
            draft_anchor_weight=draft_anchor_weight,
            fitness_contract=fitness_contract,
        )
        fitness, wins, points_for, playoff, champion = _candidate_fitness(
            state,
            candidate_team_indices,
            fitness_contract,
        )
        transaction_reward, lineup_efficiency = _candidate_auxiliary_metrics(
            state, candidate_team_indices
        )
        for policy_index in range(population_size):
            window = slice(policy_index * scenarios, (policy_index + 1) * scenarios)
            season_fitness[policy_index].append(fitness[window].mean())
            season_wins[policy_index].append(wins[window].mean())
            season_points[policy_index].append(points_for[window].mean())
            season_playoffs[policy_index].append(playoff[window].to(torch.float32).mean())
            season_champions[policy_index].append(champion[window].mean())
            season_transactions[policy_index].append(transaction_reward[window].mean())
            season_lineup_efficiency[policy_index].append(
                lineup_efficiency[window].mean()
            )

    elapsed = perf_counter() - started
    evaluations = []
    for policy_index in range(population_size):
        fitness_tensor = torch.stack(season_fitness[policy_index])
        fitness = float(fitness_tensor.mean().item())
        fitness_stddev = float(fitness_tensor.std(unbiased=False).item())
        evaluations.append(
            CudaPolicyEvaluation(
                fitness=fitness,
                fitness_stddev=fitness_stddev,
                risk_adjusted_fitness=fitness - risk_penalty * fitness_stddev,
                wins=float(torch.stack(season_wins[policy_index]).mean().item()),
                points_for=float(torch.stack(season_points[policy_index]).mean().item()),
                playoff_rate=float(torch.stack(season_playoffs[policy_index]).mean().item()),
                championship_rate=float(
                    torch.stack(season_champions[policy_index]).mean().item()
                ),
                elapsed_seconds=elapsed / population_size,
                transaction_reward=float(
                    torch.stack(season_transactions[policy_index]).mean().item()
                ),
                lineup_efficiency=float(
                    torch.stack(season_lineup_efficiency[policy_index]).mean().item()
                ),
            )
        )
    return evaluations


def clone_policy(policy: ModularManagerPolicyNetwork) -> ModularManagerPolicyNetwork:
    return copy.deepcopy(policy)


def _cpu_state_dict(policy: ModularManagerPolicyNetwork) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }


def save_cuda_training_state(
    output_path: Path,
    *,
    generation: int,
    population: list[ModularManagerPolicyNetwork],
    best_policy: ModularManagerPolicyNetwork,
    metrics: list[CudaGenerationMetrics],
    rng_state,
) -> Path:
    """Save enough state to resume the same evolutionary population."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generation": generation,
            "population": [_cpu_state_dict(policy) for policy in population],
            "best_policy": _cpu_state_dict(best_policy),
            "metrics": [metric.to_dict() for metric in metrics],
            "rng_state": rng_state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "fitness_contract": ESPN_FITNESS_CONTRACT.to_dict(),
            "fitness_contract_digest": ESPN_FITNESS_CONTRACT.digest(),
        },
        output_path,
    )
    return output_path


def validate_cuda_training_state_contract(
    state: dict,
    contract: FitnessContract = ESPN_FITNESS_CONTRACT,
) -> None:
    """Reject resumes made under different league/reward semantics."""
    expected_digest = contract.digest()
    actual_digest = state.get("fitness_contract_digest")
    if actual_digest is None:
        raise ValueError(
            "Resume checkpoint is missing fitness_contract_digest; "
            "start a new run rather than mixing objective versions."
        )
    if actual_digest != expected_digest:
        raise ValueError(
            "Resume checkpoint fitness contract does not match the active contract."
        )


def mutate_policy(
    policy: ModularManagerPolicyNetwork,
    rng: random.Random,
    strength: float,
    adapter_only: bool = True,
) -> ModularManagerPolicyNetwork:
    child = clone_policy(policy)
    with torch.no_grad():
        for name, parameter in child.named_parameters():
            if adapter_only and not (
                name.startswith("decision_heads.") or name.startswith("value_head.")
            ):
                continue
            scale = parameter.detach().float().std(unbiased=False).item() or 0.01
            noise = torch.randn_like(parameter) * (strength * scale)
            parameter.add_(noise)
    return child


def crossover_policy(
    first: ModularManagerPolicyNetwork,
    second: ModularManagerPolicyNetwork,
    rng: random.Random,
    adapter_only: bool = True,
) -> ModularManagerPolicyNetwork:
    child = clone_policy(first)
    with torch.no_grad():
        second_state = second.state_dict()
        child_state = {}
        for name, first_value in first.state_dict().items():
            if adapter_only and not (
                name.startswith("decision_heads.") or name.startswith("value_head.")
            ):
                child_state[name] = first_value.clone()
                continue
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
    scenario_bank: list[CudaSeasonState] | None = None,
    resume_state: dict | None = None,
    batch_population: bool = True,
    self_play: bool = False,
    generation_callback=None,
    checkpoint_callback=None,
    adapter_only: bool = True,
    immigrant_fraction: float = 0.10,
) -> tuple[ModularManagerPolicyNetwork, list[CudaGenerationMetrics]]:
    """Evolve neural manager policies using CUDA full-season fitness."""

    if selection_count < 1 or selection_count > population_size:
        raise ValueError("selection_count must be between one and population_size.")
    if not 0.0 <= immigrant_fraction < 1.0:
        raise ValueError("immigrant_fraction must be in [0, 1).")
    rng = random.Random(seed)
    if resume_state is None:
        torch.manual_seed(seed)
    if scenario_bank is None:
        scenario_bank = prepare_cuda_scenario_bank(
            historical_states,
            scenario_repeats=scenario_repeats,
            projection_noise=projection_noise,
            seed=seed,
        )
    population = [clone_policy(initial_policy)]
    while len(population) < population_size:
        population.append(
            mutate_policy(initial_policy, rng, mutation_strength, adapter_only=adapter_only)
        )
    best_policy = clone_policy(initial_policy)
    opponent_archive = OpponentArchive() if self_play else None
    best_risk_adjusted = float("-inf")
    metrics: list[CudaGenerationMetrics] = []
    start_generation = 1
    if resume_state is not None:
        validate_cuda_training_state_contract(resume_state, ESPN_FITNESS_CONTRACT)
        policy_device = next(initial_policy.parameters()).device
        population = []
        for state_dict in resume_state["population"]:
            policy = clone_policy(initial_policy)
            policy.load_state_dict(state_dict)
            population.append(policy.to(policy_device))
        best_policy.load_state_dict(resume_state["best_policy"])
        best_policy = best_policy.to(policy_device)
        metrics = [CudaGenerationMetrics(**item) for item in resume_state["metrics"]]
        start_generation = int(resume_state["generation"]) + 1
        if metrics:
            best_risk_adjusted = max(item.best_risk_adjusted_fitness for item in metrics)
        rng.setstate(resume_state["rng_state"])
        if "torch_rng_state" in resume_state:
            torch.set_rng_state(resume_state["torch_rng_state"])
        if (
            torch.cuda.is_available()
            and resume_state.get("cuda_rng_state_all") is not None
        ):
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])
        if len(population) != population_size:
            raise ValueError("Resume checkpoint population size does not match population_size.")
    if opponent_archive is not None:
        for index, policy in enumerate(population):
            opponent_archive.add(clone_policy(policy), label=f"initial-{index}")
    started = perf_counter()

    for generation in range(start_generation, generations + 1):
        if self_play:
            archive_opponents = (
                opponent_archive.sample(
                    max(1, historical_states[0].team_count - 1),
                    rng,
                )
                if opponent_archive is not None
                else []
            )
            evaluations = [
                evaluate_cuda_policy(
                    policy,
                    historical_states,
                    scenario_repeats=scenario_repeats,
                    projection_noise=projection_noise,
                    enable_transactions=enable_transactions,
                    seed=seed + generation * 1000,
                    draft_anchor_weight=draft_anchor_weight,
                    risk_penalty=risk_penalty,
                    compile_policy=compile_policy,
                    scenario_bank=scenario_bank,
                    opponent_policies=[*population, *archive_opponents],
                )
                for policy in population
            ]
        elif batch_population and next(population[0].parameters()).device.type == "cuda":
            evaluations = evaluate_cuda_policy_population(
                population,
                historical_states,
                scenario_bank=scenario_bank,
                scenario_repeats=scenario_repeats,
                projection_noise=projection_noise,
                enable_transactions=enable_transactions,
                seed=seed + generation * 1000,
                draft_anchor_weight=draft_anchor_weight,
                risk_penalty=risk_penalty,
            )
        else:
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
                    scenario_bank=scenario_bank,
                )
                for policy in population
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
            best_transaction_reward=best_evaluation.transaction_reward,
            best_lineup_efficiency=best_evaluation.lineup_efficiency,
        )
        metrics.append(generation_metrics)
        if opponent_archive is not None:
            generation_entries = [
                opponent_archive.add(
                    clone_policy(policy),
                    label=f"generation-{generation}-population-{index}",
                )
                for index, policy in enumerate(population)
            ]
            opponent_archive.update_entry_ratings(
                generation_entries,
                [evaluation.fitness for evaluation in evaluations],
            )
            opponent_archive.add(
                clone_policy(generation_best),
                label=f"generation-{generation}",
            )
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
            if rng.random() < immigrant_fraction:
                child = clone_policy(initial_policy)
            else:
                child = crossover_policy(first, second, rng, adapter_only=adapter_only)
            population.append(
                mutate_policy(child, rng, next_strength, adapter_only=adapter_only)
            )
        if checkpoint_callback is not None:
            checkpoint_callback(generation, population, best_policy, metrics, rng)

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
            "fitness_contract": ESPN_FITNESS_CONTRACT.to_dict(),
            "fitness_contract_digest": ESPN_FITNESS_CONTRACT.digest(),
        },
        output_path,
    )
    return output_path
