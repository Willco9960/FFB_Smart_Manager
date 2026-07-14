import copy
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import torch

from agents.neural_draft_agent import NeuralDraftAgent
from evolution.full_season import evaluate_full_season_battle_royale
from evolution.genome import DraftStrategyGenome
from evolution.population import EvaluatedAgent, rank_evaluated_agents
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from models.manager_policy_nn import ManagerPolicyNetwork
from models.weekly_projection_service import WeeklyNeuralProjectionService


@dataclass
class NeuralGenerationResult:
    generation_number: int
    best_fitness: float
    average_fitness: float
    average_wins: float
    average_points_for: float
    playoff_rate: float
    championship_count: int
    best_agent: NeuralDraftAgent
    evaluated_agents: list[EvaluatedAgent]


@dataclass
class NeuralPolicyTrainingResult:
    generations: list[NeuralGenerationResult]
    best_agent: NeuralDraftAgent


@dataclass
class NeuralPolicySeasonScenario:
    season: int
    league: League
    performances: list[WeeklyPlayerPerformance]
    synthetic_performances: list[list[WeeklyPlayerPerformance]] = field(default_factory=list)
    projection_service: WeeklyNeuralProjectionService | None = None


@dataclass(frozen=True)
class NeuralTrainingProgress:
    generation_number: int
    generation_count: int
    scenario_number: int
    scenario_count: int
    status: str
    elapsed_seconds: float
    average_fitness: float | None = None
    best_fitness: float | None = None
    average_wins: float | None = None
    average_points_for: float | None = None
    playoff_rate: float | None = None
    championship_count: int | None = None


ProgressCallback = Callable[[NeuralTrainingProgress], None]


def clone_policy_network(network: ManagerPolicyNetwork) -> ManagerPolicyNetwork:
    clone = ManagerPolicyNetwork()
    clone.load_state_dict(copy.deepcopy(network.state_dict()))
    return clone


def mutate_policy_network(
    network: ManagerPolicyNetwork,
    rng: random.Random,
    mutation_strength: float = 0.02,
) -> ManagerPolicyNetwork:
    mutated = clone_policy_network(network)

    with torch.no_grad():
        for parameter in mutated.parameters():
            noise = torch.randn(parameter.shape) * mutation_strength
            parameter.add_(noise)

    return mutated


def crossover_policy_networks(
    first: ManagerPolicyNetwork,
    second: ManagerPolicyNetwork,
    rng: random.Random,
) -> ManagerPolicyNetwork:
    child = ManagerPolicyNetwork()
    first_state = first.state_dict()
    second_state = second.state_dict()
    child_state = {}

    for name, first_value in first_state.items():
        second_value = second_state[name]
        mask = torch.rand(first_value.shape) < 0.5
        child_state[name] = torch.where(mask, first_value, second_value)

    child.load_state_dict(child_state)
    return child


def create_next_neural_generation(
    selected_agents: list[NeuralDraftAgent],
    population_size: int,
    seed: int,
    mutation_strength: float,
) -> list[NeuralDraftAgent]:
    if not selected_agents:
        raise ValueError("At least one selected neural manager is required.")

    rng = random.Random(seed)
    next_generation = list(selected_agents)

    while len(next_generation) < population_size:
        first_parent = rng.choice(selected_agents)
        second_parent = rng.choice(selected_agents)
        child_network = crossover_policy_networks(
            first_parent.policy_network,
            second_parent.policy_network,
            rng,
        )
        child_network = mutate_policy_network(
            child_network,
            rng,
            mutation_strength,
        )
        next_generation.append(
            NeuralDraftAgent(
                policy_network=child_network,
                genome=first_parent.genome,
            )
        )

    return next_generation


def aggregate_scenario_evaluations(
    agents: list[NeuralDraftAgent],
    scenario_results: list[list[EvaluatedAgent]],
) -> list[EvaluatedAgent]:
    results_by_agent: dict[int, list[EvaluatedAgent]] = {id(agent): [] for agent in agents}

    for results in scenario_results:
        for result in results:
            results_by_agent[id(result.agent)].append(result)

    aggregated_results = []

    for agent in agents:
        agent_results = results_by_agent[id(agent)]

        if not agent_results:
            raise ValueError("Every agent must have at least one scenario evaluation.")

        average_fitness = sum(result.fitness_score for result in agent_results) / len(agent_results)
        aggregated_results.append(
            replace(
                agent_results[0],
                fitness_score=round(average_fitness, 2),
            )
        )

    return aggregated_results


def evaluate_agents_across_season_scenarios(
    agents: list[NeuralDraftAgent],
    scenarios: list[NeuralPolicySeasonScenario],
    rounds: int,
    lineup_rules: tuple[LineupSlot, ...],
    seed: int,
    transaction_genome: DraftStrategyGenome,
    generation_number: int = 1,
    generation_count: int = 1,
    progress_callback: ProgressCallback | None = None,
    start_time: float | None = None,
) -> list[EvaluatedAgent]:
    scenario_results = []
    scenario_index = 0
    scenario_count = sum(1 + len(scenario.synthetic_performances) for scenario in scenarios)
    started_at = time.monotonic() if start_time is None else start_time

    for scenario in scenarios:
        performances_scenarios = [
            scenario.performances,
            *scenario.synthetic_performances,
        ]

        for performances in performances_scenarios:
            if progress_callback is not None:
                progress_callback(
                    NeuralTrainingProgress(
                        generation_number=generation_number,
                        generation_count=generation_count,
                        scenario_number=scenario_index + 1,
                        scenario_count=scenario_count,
                        status="starting",
                        elapsed_seconds=time.monotonic() - started_at,
                    )
                )

            scenario_results.append(
                evaluate_full_season_battle_royale(
                    agents=agents,
                    league=scenario.league,
                    performances=performances,
                    rounds=rounds,
                    lineup_rules=lineup_rules,
                    seed=seed + scenario_index,
                    transaction_genome_fallback=transaction_genome,
                    projection_service=scenario.projection_service,
                )
            )
            scenario_index += 1

            if progress_callback is not None:
                progress_callback(
                    NeuralTrainingProgress(
                        generation_number=generation_number,
                        generation_count=generation_count,
                        scenario_number=scenario_index,
                        scenario_count=scenario_count,
                        status="complete",
                        elapsed_seconds=time.monotonic() - started_at,
                    )
                )

    return aggregate_scenario_evaluations(agents, scenario_results)


def train_neural_policy_on_seasons(
    initial_network: ManagerPolicyNetwork,
    league: League,
    performances: list[WeeklyPlayerPerformance],
    transaction_genome: DraftStrategyGenome,
    population_size: int = 10,
    generation_count: int = 2,
    selection_count: int = 3,
    mutation_strength: float = 0.02,
    seed: int = 1,
    rounds: int = 16,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    synthetic_performances: list[list[WeeklyPlayerPerformance]] | None = None,
    projection_service: WeeklyNeuralProjectionService | None = None,
    progress_callback: ProgressCallback | None = None,
) -> NeuralPolicyTrainingResult:
    return train_neural_policy_across_seasons(
        initial_network=initial_network,
        scenarios=[
            NeuralPolicySeasonScenario(
                season=0,
                league=league,
                performances=performances,
                synthetic_performances=synthetic_performances or [],
                projection_service=projection_service,
            )
        ],
        transaction_genome=transaction_genome,
        population_size=population_size,
        generation_count=generation_count,
        selection_count=selection_count,
        mutation_strength=mutation_strength,
        seed=seed,
        rounds=rounds,
        lineup_rules=lineup_rules,
        progress_callback=progress_callback,
    )


def train_neural_policy_across_seasons(
    initial_network: ManagerPolicyNetwork,
    scenarios: list[NeuralPolicySeasonScenario],
    transaction_genome: DraftStrategyGenome,
    population_size: int = 10,
    generation_count: int = 2,
    selection_count: int = 3,
    mutation_strength: float = 0.02,
    seed: int = 1,
    rounds: int = 16,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    progress_callback: ProgressCallback | None = None,
) -> NeuralPolicyTrainingResult:
    if not scenarios:
        raise ValueError("At least one season scenario is required.")

    rng = random.Random(seed)
    agents = [
        NeuralDraftAgent(
            policy_network=clone_policy_network(initial_network),
            genome=transaction_genome,
        )
    ]

    while len(agents) < population_size:
        agents.append(
            NeuralDraftAgent(
                policy_network=mutate_policy_network(
                    initial_network,
                    rng,
                    mutation_strength,
                ),
                genome=transaction_genome,
            )
        )

    generation_results = []
    started_at = time.monotonic()

    for generation_number in range(1, generation_count + 1):
        evaluated_agents = evaluate_agents_across_season_scenarios(
            agents=agents,
            scenarios=scenarios,
            rounds=rounds,
            lineup_rules=lineup_rules,
            seed=seed + generation_number,
            transaction_genome=transaction_genome,
            generation_number=generation_number,
            generation_count=generation_count,
            progress_callback=progress_callback,
            start_time=started_at,
        )
        ranked_agents = rank_evaluated_agents(evaluated_agents)
        ranked_neural_agents = []

        for evaluated_agent in ranked_agents:
            ranked_neural_agents.append(evaluated_agent.agent)

        selected_agents = ranked_neural_agents[:selection_count]
        average_fitness = sum(
            evaluated_agent.fitness_score for evaluated_agent in evaluated_agents
        ) / len(evaluated_agents)
        average_wins = sum(
            evaluated_agent.regular_season_wins for evaluated_agent in evaluated_agents
        ) / len(evaluated_agents)
        average_points_for = sum(
            evaluated_agent.points_for for evaluated_agent in evaluated_agents
        ) / len(evaluated_agents)
        playoff_rate = sum(
            evaluated_agent.playoff_seed is not None for evaluated_agent in evaluated_agents
        ) / len(evaluated_agents)
        championship_count = sum(evaluated_agent.champion for evaluated_agent in evaluated_agents)
        generation_results.append(
            NeuralGenerationResult(
                generation_number=generation_number,
                best_fitness=ranked_agents[0].fitness_score,
                average_fitness=average_fitness,
                average_wins=average_wins,
                average_points_for=average_points_for,
                playoff_rate=playoff_rate,
                championship_count=championship_count,
                best_agent=selected_agents[0],
                evaluated_agents=evaluated_agents,
            )
        )

        if progress_callback is not None:
            progress_callback(
                NeuralTrainingProgress(
                    generation_number=generation_number,
                    generation_count=generation_count,
                    scenario_number=0,
                    scenario_count=0,
                    status="generation_complete",
                    elapsed_seconds=time.monotonic() - started_at,
                    average_fitness=generation_results[-1].average_fitness,
                    best_fitness=generation_results[-1].best_fitness,
                    average_wins=generation_results[-1].average_wins,
                    average_points_for=generation_results[-1].average_points_for,
                    playoff_rate=generation_results[-1].playoff_rate,
                    championship_count=generation_results[-1].championship_count,
                )
            )
        agents = create_next_neural_generation(
            selected_agents=selected_agents,
            population_size=population_size,
            seed=seed + generation_number,
            mutation_strength=mutation_strength,
        )

    best_agent = max(
        generation_results,
        key=lambda result: result.best_fitness,
    ).best_agent

    return NeuralPolicyTrainingResult(
        generations=generation_results,
        best_agent=best_agent,
    )
