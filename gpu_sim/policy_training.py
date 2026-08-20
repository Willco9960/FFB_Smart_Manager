"""CUDA evolutionary training for a manager policy.

This is the first training path that uses the CUDA season simulator for the
fitness loop.  A candidate policy controls one rotating team per scenario;
the other nine teams are projection-best baselines.  Waiver and trade stages
remain enabled in the tensorized season engine.
"""

from __future__ import annotations

import copy
import hashlib
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
    population_diversity: float = 0.0
    mutation_strength: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def select_training_season_indices(
    season_count: int,
    subsample_size: int,
    *,
    generation: int,
    replay_interval: int = 0,
) -> list[int]:
    """Select deterministic episodic replay seasons.

    Index zero is intentionally retained in every subsampled episode.  When
    training begins at 2000 this is the 2000 season, so the oldest reliable
    season is not silently dropped.  A positive interval performs a full-window
    replay on each interval boundary.
    """
    if season_count < 1:
        raise ValueError("season_count must be positive.")
    if subsample_size < 0 or subsample_size > season_count:
        raise ValueError("subsample_size must be zero or within season_count.")
    if generation < 1:
        raise ValueError("generation must be positive.")
    if replay_interval < 0:
        raise ValueError("replay_interval cannot be negative.")
    if not subsample_size:
        return list(range(season_count))
    if replay_interval and generation % replay_interval == 0:
        return list(range(season_count))
    if subsample_size == 1:
        return [0]
    return [
        round(index * (season_count - 1) / (subsample_size - 1))
        for index in range(subsample_size)
    ]


def summarize_cuda_throughput(
    metrics,
    *,
    population: int,
    training_seasons: int,
    scenario_repeats: int,
) -> dict[str, object]:
    """Summarize measured generation and scenario throughput.

    The stable rate is based on per-generation deltas, excluding the first
    cumulative timing sample (startup and model loading).  This is a measured
    calibration summary, not a projection from a single fast generation.
    """
    if not metrics:
        raise ValueError("At least one generation metric is required.")
    if population < 1 or training_seasons < 1 or scenario_repeats < 1:
        raise ValueError("Throughput dimensions must be positive.")
    rates = []
    previous = 0.0
    for metric in metrics:
        elapsed = float(metric.elapsed_seconds)
        delta = elapsed - previous
        if delta <= 0.0:
            raise ValueError("Generation elapsed seconds must increase strictly.")
        if previous > 0.0:
            rates.append(3600.0 / delta)
        previous = elapsed
    if not rates:
        rates = [3600.0 / previous]
    stable_gph = sum(rates) / len(rates)
    scenario_per_generation = population * training_seasons * scenario_repeats
    return {
        "elapsed_seconds": previous,
        "generations_per_hour": len(metrics) / (previous / 3600.0),
        "stable_generations_per_hour": stable_gph,
        "stable_generations_per_hour_range": [min(rates), max(rates)],
        "population_evaluations": len(metrics) * population * training_seasons,
        "scenario_evaluations": len(metrics) * scenario_per_generation,
        "normalized_scenario_evaluations_per_hour": stable_gph * scenario_per_generation,
        "population": population,
        "training_seasons": training_seasons,
        "scenario_repeats": scenario_repeats,
    }


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
        contract_digest=state.contract_digest,
        draft_floors=(
            None
            if state.draft_floors is None
            else state.draft_floors.repeat(scenario_repeats, 1)
        ),
        draft_medians=(
            None
            if state.draft_medians is None
            else state.draft_medians.repeat(scenario_repeats, 1)
        ),
        draft_ceilings=(
            None
            if state.draft_ceilings is None
            else state.draft_ceilings.repeat(scenario_repeats, 1)
        ),
        draft_boom_probabilities=None
        if state.draft_boom_probabilities is None
        else state.draft_boom_probabilities.repeat(scenario_repeats, 1),
    )


def fork_cuda_state(
    state: CudaSeasonState,
    *,
    contract_digest: str | None = None,
) -> CudaSeasonState:
    """Create a mutable simulation copy without regenerating scenarios."""
    requested_digest = state.contract_digest if contract_digest is None else contract_digest
    if state.contract_digest not in (ESPN_FITNESS_CONTRACT.digest(), requested_digest):
        raise ValueError("CUDA state fitness contract does not match the requested contract.")

    return CudaSeasonState(
        draft_projections=state.draft_projections.clone(),
        weekly_projections=state.weekly_projections.clone(),
        weekly_actual_points=state.weekly_actual_points,
        positions=state.positions,
        team_count=state.team_count,
        roster_size=state.roster_size,
        lineup_position_rules=state.lineup_position_rules,
        contract_digest=requested_digest,
        draft_floors=None if state.draft_floors is None else state.draft_floors.clone(),
        draft_medians=None if state.draft_medians is None else state.draft_medians.clone(),
        draft_ceilings=None if state.draft_ceilings is None else state.draft_ceilings.clone(),
        draft_boom_probabilities=None
        if state.draft_boom_probabilities is None
        else state.draft_boom_probabilities.clone(),
    )


def scenario_bank_digest(scenario_bank: list[CudaSeasonState]) -> str:
    """Hash immutable tensor contents used to define a scenario bank."""
    digest = hashlib.sha256()
    for season_index, state in enumerate(scenario_bank):
        digest.update(f"season-index:{season_index};".encode())
        for name, value in sorted(state.__dict__.items()):
            if not isinstance(value, torch.Tensor):
                continue
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


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
        transaction_reward = transaction_reward + (
            gains[batch, candidate_team_indices] if gains.ndim == 2 else gains
        )
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
        transaction_reward = transaction_reward + (
            gains[batch, candidate_team_indices] if gains.ndim == 2 else gains
        )
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

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.stacked_parameters = {
            name: fn(value) for name, value in self.stacked_parameters.items()
        }
        self.stacked_buffers = {
            name: fn(value) for name, value in self.stacked_buffers.items()
        }
        return self

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
    policy: ModularManagerPolicyNetwork | None,
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
    if policy is not None:
        policy.eval()
    device = historical_states[0].device
    if policy is not None:
        policy.to(device)
    scoring_policy = policy
    if compile_policy and device.type == "cuda" and policy is not None:
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
    if scenario_bank is None:
        scenario_bank = prepare_cuda_scenario_bank(
            historical_states,
            scenario_repeats=scenario_repeats,
            projection_noise=projection_noise,
            seed=seed,
        )
    templates = scenario_bank
    for template in templates:
        assignment_count = template.team_count if opponent_policies else 1
        for candidate_team in range(assignment_count):
            state = fork_cuda_state(template, contract_digest=fitness_contract.digest())
            candidate_team_indices = torch.full(
                (state.scenario_count,),
                candidate_team,
                dtype=torch.long,
                device=device,
            )
            team_policy_networks = None
            if opponent_policies:
                if policy is None:
                    raise ValueError("opponent_policies require a candidate policy.")
                archive = [opponent for opponent in opponent_policies if opponent is not policy]
                if not archive:
                    archive = [policy]
                team_policy_networks = []
                for team_index in range(state.team_count):
                    if team_index == candidate_team:
                        team_policy_networks.append(policy)
                    else:
                        archive_index = (team_index - (team_index > candidate_team)) % len(archive)
                        team_policy_networks.append(archive[archive_index])
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
            fitness_values.append(fitness.reshape(-1))
            wins_values.append(wins.reshape(-1))
            points_values.append(points_for.reshape(-1))
            playoff_values.append(playoff.to(torch.float32).reshape(-1))
            champion_values.append(champion.reshape(-1))
            transaction_values.append(transaction_reward.reshape(-1))
            lineup_efficiency_values.append(lineup_efficiency.reshape(-1))
    elapsed = perf_counter() - started
    fitness_tensor = torch.cat(fitness_values)
    fitness = float(fitness_tensor.mean().item())
    fitness_stddev = float(fitness_tensor.std(unbiased=False).item())
    return CudaPolicyEvaluation(
        fitness=fitness,
        fitness_stddev=fitness_stddev,
        risk_adjusted_fitness=fitness - (risk_penalty * fitness_stddev),
        wins=float(torch.cat(wins_values).mean().item()),
        points_for=float(torch.cat(points_values).mean().item()),
        playoff_rate=float(torch.cat(playoff_values).mean().item()),
        championship_rate=float(torch.cat(champion_values).mean().item()),
        elapsed_seconds=elapsed,
        transaction_reward=float(torch.cat(transaction_values).mean().item()),
        lineup_efficiency=float(torch.cat(lineup_efficiency_values).mean().item()),
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
    compile_policy: bool = False,
    opponent_policies: list[ModularManagerPolicyNetwork] | None = None,
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
                compile_policy=compile_policy,
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
    scoring_policy = ensemble
    if compile_policy and device.type == "cuda":
        try:
            scoring_policy = torch.compile(ensemble, mode="reduce-overhead", dynamic=False)
        except Exception as error:  # pragma: no cover - depends on local compiler
            warnings.warn(
                f"torch.compile unavailable for batched ensemble; using eager CUDA: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
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

        def repeat_optional(values: torch.Tensor | None) -> torch.Tensor | None:
            return None if values is None else values.repeat(population_size, 1)

        expected_contract_digest = fitness_contract.digest()
        if template.contract_digest not in (
            ESPN_FITNESS_CONTRACT.digest(),
            expected_contract_digest,
        ):
            raise ValueError("CUDA scenario contract does not match the requested contract.")

        state = CudaSeasonState(
            draft_projections=template.draft_projections.repeat(population_size, 1),
            weekly_projections=template.weekly_projections.repeat(population_size, 1, 1),
            weekly_actual_points=template.weekly_actual_points.repeat(population_size, 1, 1),
            positions=template.positions,
            team_count=template.team_count,
            roster_size=template.roster_size,
            lineup_position_rules=template.lineup_position_rules,
            contract_digest=expected_contract_digest,
            draft_floors=repeat_optional(template.draft_floors),
            draft_medians=repeat_optional(template.draft_medians),
            draft_ceilings=repeat_optional(template.draft_ceilings),
            draft_boom_probabilities=repeat_optional(template.draft_boom_probabilities),
        )
        if opponent_policies is None:
            candidate_team_indices = (
                torch.arange(scenarios, device=device) % state.team_count
            ).repeat(population_size)
        else:
            candidate_team_indices = torch.zeros(
                population_size * scenarios,
                dtype=torch.long,
                device=device,
            )
        team_policy_networks = None
        if opponent_policies is not None:
            if len(opponent_policies) < state.team_count - 1:
                raise ValueError("opponent_policies must cover every non-candidate team.")
            team_policy_networks = [scoring_policy]
            for opponent in opponent_policies[: state.team_count - 1]:
                opponent = opponent.to(device)
                opponent.eval()
                team_policy_networks.append(opponent)
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
            state,
            candidate_team_indices,
            fitness_contract,
        )
        transaction_reward, lineup_efficiency = _candidate_auxiliary_metrics(
            state, candidate_team_indices
        )
        for policy_index in range(population_size):
            window = slice(policy_index * scenarios, (policy_index + 1) * scenarios)
            season_fitness[policy_index].append(fitness[window].reshape(-1))
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


def _atomic_torch_save(payload: object, output_path: Path) -> None:
    """Write a checkpoint without replacing a valid file with a partial one."""
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_cuda_training_state(
    output_path: Path,
    *,
    generation: int,
    population: list[ModularManagerPolicyNetwork],
    best_policy: ModularManagerPolicyNetwork,
    metrics: list[CudaGenerationMetrics],
    rng_state,
    run_manifest: dict | None = None,
    opponent_archive: OpponentArchive | None = None,
) -> Path:
    """Save enough state to resume the same evolutionary population."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
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
            "run_manifest": copy.deepcopy(run_manifest),
            "scenario_bank_digest": (
                None if run_manifest is None else run_manifest.get("scenario_bank_digest")
            ),
            "opponent_archive": (
                None if opponent_archive is None else opponent_archive.to_state_dict()
            ),
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


def validate_cuda_training_state_manifest(
    state: dict,
    expected_manifest: dict,
) -> None:
    """Reject resumes made with different data, architecture, or search settings."""
    actual_manifest = state.get("run_manifest")
    if actual_manifest is None:
        raise ValueError(
            "Resume checkpoint is missing run_manifest; start a new run rather "
            "than mixing data or search configurations."
        )
    mismatches = {
        key: (actual_manifest.get(key), expected_manifest.get(key))
        for key in expected_manifest
        if actual_manifest.get(key) != expected_manifest.get(key)
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected!r})"
            for key, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"Resume checkpoint run manifest does not match: {details}")


def validate_cuda_policy_checkpoint_manifest(
    checkpoint: dict,
    expected_manifest: dict,
) -> None:
    """Validate identity on a published best-policy checkpoint too."""
    validate_cuda_training_state_contract(checkpoint)
    validate_cuda_training_state_manifest(checkpoint, expected_manifest)


def mutate_policy(
    policy: ModularManagerPolicyNetwork,
    rng: random.Random,
    strength: float,
    adapter_only: bool = True,
) -> ModularManagerPolicyNetwork:
    child = clone_policy(policy)
    with torch.no_grad():
        for name, parameter in child.named_parameters():
            if name.startswith("value_head.") or (
                adapter_only and not name.startswith("decision_heads.")
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
            if name.startswith("value_head.") or (
                adapter_only and not name.startswith("decision_heads.")
            ):
                child_state[name] = first_value.clone()
                continue
            mask = torch.rand(first_value.shape, device=first_value.device) < 0.5
            child_state[name] = torch.where(mask, first_value, second_state[name])
        child.load_state_dict(child_state)
    return child


def policy_population_diversity(
    policies: list[ModularManagerPolicyNetwork],
    *,
    adapter_only: bool = True,
) -> float:
    """Return mean normalized parameter distance across a population."""
    if len(policies) < 2:
        return 0.0
    vectors = []
    for policy in policies:
        values = [
            parameter.detach().float().reshape(-1)
            for name, parameter in policy.named_parameters()
            if not adapter_only or name.startswith("decision_heads.")
        ]
        vectors.append(torch.cat(values))
    stacked = torch.stack(vectors)
    distances = torch.pdist(stacked)
    scale = stacked.std(dim=0, unbiased=False).mean().clamp_min(1e-6)
    return float((distances.mean() / scale).item())


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
    exact_policy_head_parity: bool = True,
    self_play: bool = False,
    opponent_archive_size: int = 64,
    self_play_interval: int = 1,
    generation_callback=None,
    checkpoint_callback=None,
    adapter_only: bool = True,
    immigrant_fraction: float = 0.10,
    scenario_refresh_generations: int = 0,
    season_subsample_size: int = 0,
    season_replay_interval: int = 0,
    require_complete_fitness_contract: bool = False,
    run_manifest: dict | None = None,
) -> tuple[ModularManagerPolicyNetwork, list[CudaGenerationMetrics]]:
    """Evolve neural manager policies using CUDA full-season fitness."""

    if require_complete_fitness_contract and enable_transactions:
        unsupported = {
            "replacement_value_weight": ESPN_FITNESS_CONTRACT.replacement_value_weight,
            "invalid_action_penalty": ESPN_FITNESS_CONTRACT.invalid_action_penalty,
        }
        active_unsupported = {
            name: value for name, value in unsupported.items() if value != 0.0
        }
        if active_unsupported:
            raise ValueError(
                "CUDA fitness contract is incomplete for promotion: "
                + ", ".join(active_unsupported)
            )

    if selection_count < 1 or selection_count > population_size:
        raise ValueError("selection_count must be between one and population_size.")
    if not 0.0 <= immigrant_fraction < 1.0:
        raise ValueError("immigrant_fraction must be in [0, 1).")
    if scenario_refresh_generations < 0:
        raise ValueError("scenario_refresh_generations cannot be negative.")
    if opponent_archive_size < 1:
        raise ValueError("opponent_archive_size must be positive.")
    if self_play_interval < 1:
        raise ValueError("self_play_interval must be positive.")
    if season_subsample_size < 0 or season_subsample_size > len(historical_states):
        raise ValueError("season_subsample_size must be zero or within the training-state count.")
    if season_replay_interval < 0:
        raise ValueError("season_replay_interval cannot be negative.")
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
    opponent_archive = None
    archive_restored = False
    if self_play:
        opponent_archive_state = (
            None if resume_state is None else resume_state.get("opponent_archive")
        )
        if opponent_archive_state is not None:
            opponent_archive = OpponentArchive.from_state_dict(
                opponent_archive_state,
                initial_policy,
                next(initial_policy.parameters()).device,
            )
            archive_restored = True
            if opponent_archive.max_size != opponent_archive_size:
                raise ValueError(
                    "Resume checkpoint opponent archive size does not match opponent_archive_size."
                )
        else:
            opponent_archive = OpponentArchive(max_size=opponent_archive_size)
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
    if opponent_archive is not None and not archive_restored:
        for index, policy in enumerate(population):
            opponent_archive.add(clone_policy(policy), label=f"initial-{index}")
    active_scenario_generation = 1
    if scenario_refresh_generations and start_generation > 1:
        active_scenario_generation = (
            ((start_generation - 1) // scenario_refresh_generations)
            * scenario_refresh_generations
            + 1
        )
        if active_scenario_generation > 1:
            scenario_bank = prepare_cuda_scenario_bank(
                historical_states,
                scenario_repeats=scenario_repeats,
                projection_noise=projection_noise,
                seed=seed + active_scenario_generation * 100_000,
            )
    expected_scenario_digest = (
        None if resume_state is None else resume_state.get("scenario_bank_digest")
    )
    if expected_scenario_digest is not None:
        actual_scenario_digest = scenario_bank_digest(scenario_bank)
        if actual_scenario_digest != expected_scenario_digest:
            raise ValueError(
                "Resume checkpoint scenario bank digest does not match reconstructed scenarios."
            )
    started = perf_counter()
    resumed_elapsed_seconds = metrics[-1].elapsed_seconds if metrics else 0.0

    for generation in range(start_generation, generations + 1):
        active_states = historical_states
        active_scenario_bank = scenario_bank
        season_indices = list(range(len(historical_states)))
        if season_subsample_size:
            season_indices = select_training_season_indices(
                len(historical_states),
                season_subsample_size,
                generation=generation,
                replay_interval=season_replay_interval,
            )
            active_states = [historical_states[index] for index in season_indices]
            active_scenario_bank = [scenario_bank[index] for index in season_indices]
        if scenario_refresh_generations:
            requested_scenario_generation = (
                ((generation - 1) // scenario_refresh_generations)
                * scenario_refresh_generations
                + 1
            )
            if requested_scenario_generation != active_scenario_generation:
                active_scenario_generation = requested_scenario_generation
                scenario_bank = prepare_cuda_scenario_bank(
                    historical_states,
                    scenario_repeats=scenario_repeats,
                    projection_noise=projection_noise,
                    seed=seed + active_scenario_generation * 100_000,
                )
                active_scenario_bank = [scenario_bank[index] for index in season_indices]
        run_self_play = self_play and generation % self_play_interval == 0
        if run_self_play:
            archive_opponents = (
                opponent_archive.sample(
                    max(1, active_states[0].team_count - 1),
                    rng,
                )
                if opponent_archive is not None
                else []
            )
            evaluations = evaluate_cuda_policy_population(
                population,
                active_states,
                scenario_bank=active_scenario_bank,
                scenario_repeats=scenario_repeats,
                projection_noise=projection_noise,
                enable_transactions=enable_transactions,
                seed=seed + generation * 1000,
                draft_anchor_weight=draft_anchor_weight,
                risk_penalty=risk_penalty,
                exact_policy_head_parity=exact_policy_head_parity,
                compile_policy=compile_policy,
                opponent_policies=archive_opponents,
            )
        elif batch_population and next(population[0].parameters()).device.type == "cuda":
            evaluations = evaluate_cuda_policy_population(
                population,
                active_states,
                scenario_bank=active_scenario_bank,
                scenario_repeats=scenario_repeats,
                projection_noise=projection_noise,
                enable_transactions=enable_transactions,
                seed=seed + generation * 1000,
                draft_anchor_weight=draft_anchor_weight,
                risk_penalty=risk_penalty,
                exact_policy_head_parity=exact_policy_head_parity,
                compile_policy=compile_policy,
            )
        else:
            evaluations = [
                evaluate_cuda_policy(
                    policy,
                    active_states,
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
        next_strength = mutation_strength - (
            (mutation_strength - final_mutation_strength)
            * ((generation - 1) / max(generations - 1, 1))
        )
        diversity = policy_population_diversity(population, adapter_only=adapter_only)
        elapsed = resumed_elapsed_seconds + perf_counter() - started
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
            population_diversity=diversity,
            mutation_strength=next_strength,
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
        population = [clone_policy(policy) for policy in selected[: min(2, len(selected))]]
        adaptive_strength = next_strength * (1.25 if diversity < 1.0 else 1.0)
        while len(population) < population_size:
            first = rng.choice(selected)
            second = rng.choice(selected)
            if rng.random() < immigrant_fraction:
                child = clone_policy(initial_policy)
            else:
                child = crossover_policy(first, second, rng, adapter_only=adapter_only)
            population.append(
                mutate_policy(child, rng, adaptive_strength, adapter_only=adapter_only)
            )
        if checkpoint_callback is not None:
            if isinstance(run_manifest, dict):
                run_manifest["scenario_bank_digest"] = scenario_bank_digest(active_scenario_bank)
            checkpoint_callback(generation, population, best_policy, metrics, rng, opponent_archive)

    return best_policy, metrics


def save_cuda_policy_checkpoint(
    policy: ModularManagerPolicyNetwork,
    output_path: Path,
    metrics: list[CudaGenerationMetrics],
    run_manifest: dict | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
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
            "run_manifest": copy.deepcopy(run_manifest),
        },
        output_path,
    )
    return output_path
