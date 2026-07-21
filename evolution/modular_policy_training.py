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
from evolution.full_season import evaluate_full_season_battle_royale
from evolution.genome import DraftStrategyGenome
from evolution.population import EvaluatedAgent
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from models.modular_manager_policy import ModularManagerPolicyNetwork


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

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


GenerationCallback = Callable[[ModularGenerationMetrics, ModularManagerPolicyNetwork], None]


def select_scenarios_for_generation(
    scenarios: list[tuple[League, list[WeeklyPlayerPerformance]]],
    generation_number: int,
    scenarios_per_generation: int | None = None,
    full_evaluation_interval: int = 0,
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

    start_index = ((generation_number - 1) * scenarios_per_generation) % len(scenarios)
    return [
        scenarios[(start_index + offset) % len(scenarios)]
        for offset in range(scenarios_per_generation)
    ]


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
            parameter.add_(torch.randn(parameter.shape, generator=generator) * mutation_strength)
    return mutated


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
) -> tuple[ModularManagerPolicyNetwork, list[float]]:
    if population_size < selection_count or selection_count < 1:
        raise ValueError("selection_count must be between one and population_size.")
    if not scenarios:
        raise ValueError("At least one scenario is required.")

    rng = random.Random(seed)
    population = [clone_modular_policy(initial_policy)]
    while len(population) < population_size:
        population.append(mutate_modular_policy(initial_policy, rng, mutation_strength))

    history = []
    best_policy = clone_modular_policy(initial_policy)
    best_score = float("-inf")
    training_started = perf_counter()
    for generation in range(generations):
        generation_scenarios = select_scenarios_for_generation(
            scenarios=scenarios,
            generation_number=generation + 1,
            scenarios_per_generation=scenarios_per_generation,
            full_evaluation_interval=full_evaluation_interval,
        )
        neural_agents = [
            NeuralDraftAgent(policy_network=policy, genome=transaction_genome)
            for policy in population
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
            average = sum(result.fitness_score for result in agent_results) / len(agent_results)
            averaged.append((average, agent))
        averaged.sort(key=lambda item: item[0], reverse=True)
        generation_best_score = averaged[0][0]
        neural_scores = [score for score, _ in averaged]
        neural_agent_ids = {id(agent) for agent in neural_agents}
        baseline_results = [
            result for result in results if id(result.agent) not in neural_agent_ids
        ]
        best_agent = averaged[0][1]
        best_agent_results = results_by_agent[id(best_agent)]

        history.append(round(generation_best_score, 2))
        if generation_best_score > best_score:
            best_score = generation_best_score
            best_policy = clone_modular_policy(best_agent.policy_network)

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
            mutation_strength=round(mutation_strength / (generation + 1), 6),
            elapsed_seconds=round(perf_counter() - training_started, 2),
            cumulative_best_fitness=round(best_score, 2),
        )
        if generation_callback is not None:
            generation_callback(metrics, best_policy)

        selected = [agent for _, agent in averaged[:selection_count]]
        # Carry policy networks into the next generation.  Keeping the
        # NeuralDraftAgent wrapper here would nest agents inside agents on the
        # next loop, so the draft agent would eventually receive an object
        # without score_action/score_decisions methods.
        population = [agent.policy_network for agent in selected]
        while len(population) < population_size:
            first = rng.choice(selected)
            second = rng.choice(selected)
            child = crossover_modular_policies(first.policy_network, second.policy_network, rng)
            population.append(
                mutate_modular_policy(child, rng, mutation_strength / (generation + 1))
            )

    return best_policy, history
