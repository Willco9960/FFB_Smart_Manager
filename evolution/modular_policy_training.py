"""Evolutionary self-play trainer for the modular manager policy.

Evolution is retained for population diversity and opponent adaptation, but it
mutates the modular policy after projection/behavioral pretraining rather than
trying to learn player projections from championship rewards alone.
"""

import copy
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from statistics import mean, median, pstdev
from time import perf_counter

import torch

from agents.baseline_agents import create_baseline_opponents
from agents.neural_draft_agent import NeuralDraftAgent
from evolution.full_season import (
    TRANSACTION_MODES,
    TransactionMode,
    evaluate_full_season_battle_royale,
)
from evolution.genome import DraftStrategyGenome
from evolution.population import EvaluatedAgent
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from models.modular_manager_policy import ModularManagerPolicyNetwork
from models.transaction_value import TransactionValueNetwork


@dataclass(frozen=True)
class ModularGenerationMetrics:
    """Inspectable metrics emitted after each self-play generation."""

    generation_number: int
    generation_count: int
    scenario_count: int
    neural_population: int
    baseline_population: int
    best_fitness: float
    average_fitness: float
    median_fitness: float
    fitness_stddev: float
    best_wins: float
    best_points_for: float
    best_playoff_rate: float
    best_championship_rate: float
    best_transaction_reward: float
    baseline_average_fitness: float | None
    baseline_best_fitness: float | None
    mutation_strength: float
    elapsed_seconds: float
    cumulative_best_fitness: float
    cumulative_best_generation: int
    scenario_labels: tuple[str, ...]
    best_risk_adjusted_fitness: float | None = None
    policy_population_diversity: float = 0.0

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


GenerationCallback = Callable[[ModularGenerationMetrics, ModularManagerPolicyNetwork], None]
FinalEvaluationCallback = Callable[[dict, ModularManagerPolicyNetwork], None]


def select_scenarios_for_generation(
    scenarios: list[tuple[League, list[WeeklyPlayerPerformance]]],
    generation_number: int,
    scenarios_per_generation: int | None = None,
    full_evaluation_interval: int = 0,
    anchor_scenarios_per_generation: int = 4,
) -> list[tuple[League, list[WeeklyPlayerPerformance]]]:
    """Select a deterministic rotating subset without changing the default.

    A full evaluation is retained whenever the requested interval is reached.
    This lets long runs spend most generations exploring quickly while still
    checking the entire historical library periodically.
    """

    if not scenarios:
        return []
    if scenarios_per_generation is None or scenarios_per_generation <= 0:
        return list(scenarios)
    if scenarios_per_generation >= len(scenarios):
        return list(scenarios)
    if full_evaluation_interval > 0 and generation_number % full_evaluation_interval == 0:
        return list(scenarios)

    anchor_count = min(anchor_scenarios_per_generation, scenarios_per_generation)
    if anchor_count <= 0:
        anchor_indices: list[int] = []
    elif anchor_count == 1:
        anchor_indices = [0]
    else:
        anchor_indices = sorted(
            {
                round(index * (len(scenarios) - 1) / (anchor_count - 1))
                for index in range(anchor_count)
            }
        )

    rotating_count = scenarios_per_generation - len(anchor_indices)
    rotating_indices = [index for index in range(len(scenarios)) if index not in anchor_indices]
    start_index = ((generation_number - 1) * rotating_count) % len(rotating_indices)
    selected_indices = anchor_indices + [
        rotating_indices[(start_index + offset) % len(rotating_indices)]
        for offset in range(rotating_count)
    ]
    return [scenarios[index] for index in sorted(selected_indices)]


def clone_modular_policy(network: ModularManagerPolicyNetwork) -> ModularManagerPolicyNetwork:
    clone = ModularManagerPolicyNetwork(
        player_feature_count=network.player_feature_count,
        state_feature_count=network.state_feature_count,
        hidden_size=network.hidden_size,
    )
    clone.load_state_dict(copy.deepcopy(network.state_dict()))
    return clone


def mutate_modular_policy(
    network: ModularManagerPolicyNetwork,
    rng: random.Random,
    mutation_strength: float = 0.01,
) -> ModularManagerPolicyNetwork:
    mutated = clone_modular_policy(network)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(rng.randrange(0, 2**63 - 1))
    with torch.no_grad():
        for parameter in mutated.parameters():
            # A non-zero scale floor prevents the population from freezing when
            # several generations converge to nearly identical weights.
            parameter_scale = max(float(parameter.detach().std(unbiased=False).item()), 0.05)
            parameter.add_(
                torch.randn(parameter.shape, generator=generator)
                * mutation_strength
                * parameter_scale
            )
    return mutated


def calculate_policy_population_diversity(
    population: list[ModularManagerPolicyNetwork],
) -> float:
    """Return mean normalized pairwise parameter distance for telemetry."""

    if len(population) < 2:
        return 0.0

    distances = []
    for index, first in enumerate(population):
        first_parameters = list(first.parameters())
        for second in population[index + 1 :]:
            second_parameters = list(second.parameters())
            numerator = sum(
                float((left.detach() - right.detach()).abs().mean().item())
                for left, right in zip(first_parameters, second_parameters, strict=True)
            )
            denominator = sum(
                max(float(parameter.detach().abs().mean().item()), 0.01)
                for parameter in first_parameters
            )
            distances.append(numerator / denominator)

    return round(mean(distances), 6) if distances else 0.0


def adapt_mutation_for_diversity(
    mutation_strength: float,
    population_diversity: float,
    diversity_floor: float,
    diversity_boost: float,
) -> float:
    """Increase exploration when the population is collapsing."""

    if diversity_floor <= 0.0 or population_diversity >= diversity_floor:
        return mutation_strength
    if diversity_boost < 1.0:
        raise ValueError("diversity_boost must be at least one.")

    return round(mutation_strength * diversity_boost, 6)


def calculate_generation_mutation_strength(
    generation_number: int,
    generation_count: int,
    initial_mutation_strength: float,
    final_mutation_strength: float,
) -> float:
    """Linearly anneal mutation while retaining a useful exploration floor."""

    if generation_count <= 1:
        return round(final_mutation_strength, 6)

    progress = (generation_number - 1) / (generation_count - 1)
    return round(
        initial_mutation_strength
        + (final_mutation_strength - initial_mutation_strength) * progress,
        6,
    )


def calculate_risk_adjusted_fitness(
    scores: list[float],
    risk_penalty: float,
) -> float:
    """Prefer policies that score well without relying on one lucky season."""

    if not scores:
        raise ValueError("At least one score is required.")

    spread = pstdev(scores) if len(scores) > 1 else 0.0
    return mean(scores) - (risk_penalty * spread)


def crossover_modular_policies(
    first: ModularManagerPolicyNetwork,
    second: ModularManagerPolicyNetwork,
    rng: random.Random,
) -> ModularManagerPolicyNetwork:
    child = clone_modular_policy(first)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(rng.randrange(0, 2**63 - 1))
    child_state = {}
    for name, first_value in first.state_dict().items():
        second_value = second.state_dict()[name]
        mask = torch.rand(first_value.shape, generator=generator) < 0.5
        child_state[name] = torch.where(mask, first_value, second_value)
    child.load_state_dict(child_state)
    return child


def train_modular_policy_self_play(
    initial_policy: ModularManagerPolicyNetwork,
    scenarios: list[tuple[League, list[WeeklyPlayerPerformance]]],
    transaction_genome: DraftStrategyGenome,
    population_size: int = 10,
    generations: int = 5,
    selection_count: int = 3,
    mutation_strength: float = 0.01,
    seed: int = 1,
    rounds: int = 16,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    include_baseline_opponents: bool = True,
    generation_callback: GenerationCallback | None = None,
    scenarios_per_generation: int | None = None,
    full_evaluation_interval: int = 0,
    anchor_scenarios_per_generation: int = 4,
    final_selection_count: int = 3,
    risk_penalty: float = 0.10,
    final_mutation_strength: float | None = None,
    elite_count: int = 1,
    draft_exploration_rate: float = 0.04,
    draft_exploration_top_k: int = 5,
    diversity_floor: float = 0.002,
    diversity_mutation_boost: float = 1.5,
    transaction_ablation: bool = False,
    transaction_mode: TransactionMode = "hybrid",
    transaction_value_model: TransactionValueNetwork | None = None,
    final_evaluation_callback: FinalEvaluationCallback | None = None,
) -> tuple[ModularManagerPolicyNetwork, list[float]]:
    if population_size < selection_count or selection_count < 1:
        raise ValueError("selection_count must be between one and population_size.")
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    if final_selection_count < 1:
        raise ValueError("final_selection_count must be at least one.")
    if risk_penalty < 0.0:
        raise ValueError("risk_penalty cannot be negative.")
    if elite_count < 0 or elite_count > population_size:
        raise ValueError("elite_count must be between zero and population_size.")
    if not 0.0 <= draft_exploration_rate <= 1.0:
        raise ValueError("draft_exploration_rate must be between zero and one.")
    if draft_exploration_top_k < 1:
        raise ValueError("draft_exploration_top_k must be at least one.")
    if diversity_floor < 0.0:
        raise ValueError("diversity_floor cannot be negative.")
    if diversity_mutation_boost < 1.0:
        raise ValueError("diversity_mutation_boost must be at least one.")
    if transaction_mode not in TRANSACTION_MODES:
        raise ValueError(f"Unknown transaction mode: {transaction_mode}")
    if final_mutation_strength is None:
        final_mutation_strength = mutation_strength * 0.25

    rng = random.Random(seed)
    population = [clone_modular_policy(initial_policy)]
    while len(population) < population_size:
        population.append(mutate_modular_policy(initial_policy, rng, mutation_strength))

    history = []
    best_policy = clone_modular_policy(initial_policy)
    best_score = float("-inf")
    best_risk_adjusted_score = float("-inf")
    best_generation = 0
    candidate_policies: list[tuple[int, float, float, ModularManagerPolicyNetwork]] = []
    training_started = perf_counter()
    for generation in range(generations):
        generation_scenarios = select_scenarios_for_generation(
            scenarios=scenarios,
            generation_number=generation + 1,
            scenarios_per_generation=scenarios_per_generation,
            full_evaluation_interval=full_evaluation_interval,
            anchor_scenarios_per_generation=anchor_scenarios_per_generation,
        )
        neural_agents = [
            NeuralDraftAgent(
                policy_network=policy,
                genome=transaction_genome,
                exploration_rate=draft_exploration_rate,
                exploration_top_k=draft_exploration_top_k,
                random_seed=seed + generation * population_size + index,
            )
            for index, policy in enumerate(population)
        ]
        agents = list(neural_agents)
        if include_baseline_opponents:
            minimum_total = len(neural_agents) + 10
            total_agent_count = ((minimum_total + 9) // 10) * 10
            agents.extend(
                create_baseline_opponents(
                    opponent_count=total_agent_count - len(neural_agents),
                    seed=seed + generation,
                )
            )
        results: list[EvaluatedAgent] = []
        for scenario_index, (league, performances) in enumerate(generation_scenarios):
            results.extend(
                evaluate_full_season_battle_royale(
                    agents=agents,
                    league=league,
                    performances=performances,
                    rounds=rounds,
                    lineup_rules=lineup_rules,
                    seed=seed + generation * 1000 + scenario_index,
                    transaction_genome_fallback=transaction_genome,
                    transaction_mode=transaction_mode,
                    transaction_value_model=transaction_value_model,
                )
            )

        results_by_agent: dict[int, list[EvaluatedAgent]] = {
            id(agent): [] for agent in neural_agents
        }
        for result in results:
            if id(result.agent) in results_by_agent:
                results_by_agent[id(result.agent)].append(result)
        averaged = []
        for agent in neural_agents:
            agent_results = results_by_agent[id(agent)]
            scores = [result.fitness_score for result in agent_results]
            average = mean(scores)
            risk_adjusted = calculate_risk_adjusted_fitness(scores, risk_penalty)
            averaged.append(
                (risk_adjusted, average, pstdev(scores) if len(scores) > 1 else 0.0, agent)
            )
        averaged.sort(key=lambda item: (item[0], item[1]), reverse=True)
        generation_best_risk_adjusted = averaged[0][0]
        generation_best_score = averaged[0][1]
        neural_scores = [average for _, average, _, _ in averaged]
        neural_agent_ids = {id(agent) for agent in neural_agents}
        baseline_results = [
            result for result in results if id(result.agent) not in neural_agent_ids
        ]
        best_agent = averaged[0][3]
        best_agent_results = results_by_agent[id(best_agent)]
        population_diversity = calculate_policy_population_diversity(population)

        history.append(round(generation_best_score, 2))
        if generation_best_risk_adjusted > best_risk_adjusted_score:
            best_risk_adjusted_score = generation_best_risk_adjusted
            best_score = generation_best_score
            best_policy = clone_modular_policy(best_agent.policy_network)
            best_generation = generation + 1

        candidate_policies.append(
            (
                generation + 1,
                generation_best_score,
                generation_best_risk_adjusted,
                clone_modular_policy(best_agent.policy_network),
            )
        )

        current_mutation_strength = calculate_generation_mutation_strength(
            generation_number=generation + 1,
            generation_count=generations,
            initial_mutation_strength=mutation_strength,
            final_mutation_strength=final_mutation_strength,
        )
        metrics = ModularGenerationMetrics(
            generation_number=generation + 1,
            generation_count=generations,
            scenario_count=len(generation_scenarios),
            neural_population=len(neural_agents),
            baseline_population=len(agents) - len(neural_agents),
            best_fitness=round(generation_best_score, 2),
            average_fitness=round(mean(neural_scores), 2),
            median_fitness=round(median(neural_scores), 2),
            fitness_stddev=round(pstdev(neural_scores), 2) if len(neural_scores) > 1 else 0.0,
            best_wins=round(mean(result.regular_season_wins for result in best_agent_results), 2),
            best_points_for=round(mean(result.points_for for result in best_agent_results), 2),
            best_playoff_rate=round(mean(result.playoff_rate for result in best_agent_results), 4),
            best_championship_rate=round(
                mean(result.championship_rate for result in best_agent_results),
                4,
            ),
            best_transaction_reward=round(
                mean(result.transaction_reward for result in best_agent_results),
                2,
            ),
            baseline_average_fitness=(
                round(mean(result.fitness_score for result in baseline_results), 2)
                if baseline_results
                else None
            ),
            baseline_best_fitness=(
                round(max(result.fitness_score for result in baseline_results), 2)
                if baseline_results
                else None
            ),
            mutation_strength=current_mutation_strength,
            elapsed_seconds=round(perf_counter() - training_started, 2),
            cumulative_best_fitness=round(best_score, 2),
            cumulative_best_generation=best_generation,
            scenario_labels=tuple(league.name for league, _ in generation_scenarios),
            best_risk_adjusted_fitness=round(generation_best_risk_adjusted, 2),
            policy_population_diversity=population_diversity,
        )
        if generation_callback is not None:
            generation_callback(metrics, best_policy)

        selected = [agent for _, _, _, agent in averaged[:selection_count]]
        # Carry policy networks into the next generation.  Keeping the
        # NeuralDraftAgent wrapper here would nest agents inside agents on the
        # next loop, so the draft agent would eventually receive an object
        # without score_action/score_decisions methods.
        population = []
        for _ in range(elite_count):
            population.append(clone_modular_policy(best_policy))
        population.extend(agent.policy_network for agent in selected)
        population = population[:population_size]
        next_mutation_strength = calculate_generation_mutation_strength(
            generation_number=generation + 2,
            generation_count=generations,
            initial_mutation_strength=mutation_strength,
            final_mutation_strength=final_mutation_strength,
        )
        next_mutation_strength = adapt_mutation_for_diversity(
            mutation_strength=next_mutation_strength,
            population_diversity=population_diversity,
            diversity_floor=diversity_floor,
            diversity_boost=diversity_mutation_boost,
        )
        while len(population) < population_size:
            first = rng.choice(selected)
            second = rng.choice(selected)
            child = crossover_modular_policies(first.policy_network, second.policy_network, rng)
            population.append(mutate_modular_policy(child, rng, next_mutation_strength))

    final_candidates = sorted(
        candidate_policies,
        key=lambda item: (item[2], item[1]),
        reverse=True,
    )[: min(final_selection_count, len(candidate_policies))]
    final_evaluations = []
    for generation_number, training_fitness, training_risk_adjusted, policy in final_candidates:
        candidate = NeuralDraftAgent(policy_network=policy, genome=transaction_genome)
        opponents = create_baseline_opponents(
            opponent_count=len(scenarios[0][0].teams) - 1,
            seed=seed + 100_000,
        )

        def evaluate_transaction_mode(
            mode: TransactionMode,
            candidate_policy=candidate,
            opponent_agents=opponents,
        ):
            candidate_results: list[EvaluatedAgent] = []
            for scenario_index, (league, performances) in enumerate(scenarios):
                results = evaluate_full_season_battle_royale(
                    agents=[candidate_policy, *opponent_agents],
                    league=league,
                    performances=performances,
                    rounds=rounds,
                    lineup_rules=lineup_rules,
                    # Common random numbers make candidate comparisons fair: a
                    # candidate must win because of its decisions, not because it
                    # received a friendlier draft shuffle or matchup schedule.
                    seed=seed + 200_000 + scenario_index,
                    transaction_genome_fallback=transaction_genome,
                    transaction_mode=mode,
                    transaction_value_model=transaction_value_model,
                )
                candidate_results.append(
                    next(result for result in results if result.agent is candidate_policy)
                )
            return candidate_results

        candidate_results = evaluate_transaction_mode(transaction_mode)

        transaction_arms = {}
        if transaction_ablation:
            for mode in TRANSACTION_MODES:
                if mode == transaction_mode:
                    continue
                arm_results = evaluate_transaction_mode(mode)
                arm_fitness = [result.fitness_score for result in arm_results]
                transaction_arms[mode] = {
                    "fitness": round(mean(arm_fitness), 2),
                    "fitness_stddev": round(pstdev(arm_fitness), 2),
                    "wins": round(mean(result.regular_season_wins for result in arm_results), 2),
                    "playoff_rate": round(mean(result.playoff_rate for result in arm_results), 4),
                    "championship_rate": round(
                        mean(result.championship_rate for result in arm_results),
                        4,
                    ),
                    "transaction_reward": round(
                        mean(result.transaction_reward for result in arm_results),
                        2,
                    ),
                }

        final_evaluations.append(
            {
                "generation_number": generation_number,
                "training_fitness": round(training_fitness, 2),
                "training_risk_adjusted_fitness": round(training_risk_adjusted, 2),
                "full_evaluation_fitness": round(
                    mean(result.fitness_score for result in candidate_results),
                    2,
                ),
                "full_evaluation_fitness_stddev": round(
                    pstdev(result.fitness_score for result in candidate_results),
                    2,
                ),
                "wins": round(mean(result.regular_season_wins for result in candidate_results), 2),
                "points_for": round(mean(result.points_for for result in candidate_results), 2),
                "playoff_rate": round(mean(result.playoff_rate for result in candidate_results), 4),
                "championship_rate": round(
                    mean(result.championship_rate for result in candidate_results),
                    4,
                ),
                "transaction_reward": round(
                    mean(result.transaction_reward for result in candidate_results),
                    2,
                ),
                "transaction_mode": transaction_mode,
                "transaction_ablation": transaction_arms,
            }
        )

    for evaluation in final_evaluations:
        evaluation["risk_adjusted_fitness"] = round(
            evaluation["full_evaluation_fitness"]
            - risk_penalty * evaluation["full_evaluation_fitness_stddev"],
            2,
        )

    selected_final_index = max(
        range(len(final_evaluations)),
        key=lambda index: (
            final_evaluations[index]["risk_adjusted_fitness"],
            final_evaluations[index]["full_evaluation_fitness"],
        ),
    )
    selected_generation, _, _, selected_policy = final_candidates[selected_final_index]
    best_policy = clone_modular_policy(selected_policy)
    final_report = {
        "selected_generation": selected_generation,
        "candidates": final_evaluations,
        "evaluation_scenario_count": len(scenarios),
    }
    if final_evaluation_callback is not None:
        final_evaluation_callback(final_report, best_policy)

    return best_policy, history
