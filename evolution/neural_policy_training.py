import copy
import random
import statistics
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
    championship_count: float
    championship_rate: float
    average_fitness_stddev: float
    best_risk_adjusted_fitness: float
    best_agent_average_wins: float
    best_agent_average_points_for: float
    best_agent_playoff_rate: float
    best_agent_championship_rate: float
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
    championship_count: float | None = None
    championship_rate: float | None = None
    average_fitness_stddev: float | None = None
    best_risk_adjusted_fitness: float | None = None
    best_agent_average_wins: float | None = None
    best_agent_average_points_for: float | None = None
    best_agent_playoff_rate: float | None = None
    best_agent_championship_rate: float | None = None


ProgressCallback = Callable[[NeuralTrainingProgress], None]
DEFAULT_CONSISTENCY_PENALTY = 0.25


def clone_policy_network(network: ManagerPolicyNetwork) -> ManagerPolicyNetwork:
    clone = ManagerPolicyNetwork()
    clone.load_state_dict(copy.deepcopy(network.state_dict()))
    return clone


def mutate_policy_network(
    network: ManagerPolicyNetwork,
    rng: random.Random,
    mutation_strength: float = 0.02,
    torch_generator: torch.Generator | None = None,
) -> ManagerPolicyNetwork:
    mutated = clone_policy_network(network)

    if torch_generator is None:
        first_parameter = next(mutated.parameters())
        torch_generator = torch.Generator(device=first_parameter.device)
        torch_generator.manual_seed(rng.randrange(0, 2**63 - 1))

    with torch.no_grad():
        for parameter in mutated.parameters():
            noise = torch.randn(
                parameter.shape,
                generator=torch_generator,
                device=parameter.device,
                dtype=parameter.dtype,
            ) * mutation_strength
            parameter.add_(noise)

    return mutated


def crossover_policy_networks(
    first: ManagerPolicyNetwork,
    second: ManagerPolicyNetwork,
    rng: random.Random,
    torch_generator: torch.Generator | None = None,
) -> ManagerPolicyNetwork:
    child = ManagerPolicyNetwork()
    first_state = first.state_dict()
    second_state = second.state_dict()
    child_state = {}

    if torch_generator is None:
        first_parameter = next(first.parameters())
        torch_generator = torch.Generator(device=first_parameter.device)
        torch_generator.manual_seed(rng.randrange(0, 2**63 - 1))

    for name, first_value in first_state.items():
        second_value = second_state[name]
        mask = torch.rand(
            first_value.shape,
            generator=torch_generator,
            device=first_value.device,
        ) < 0.5
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
    first_parameter = next(selected_agents[0].policy_network.parameters())
    torch_generator = torch.Generator(device=first_parameter.device)
    torch_generator.manual_seed(seed)

    while len(next_generation) < population_size:
        first_parent = rng.choice(selected_agents)
        second_parent = rng.choice(selected_agents)
        child_network = crossover_policy_networks(
            first_parent.policy_network,
            second_parent.policy_network,
            rng,
            torch_generator,
        )
        child_network = mutate_policy_network(
            child_network,
            rng,
            mutation_strength,
            torch_generator,
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
    consistency_penalty: float = DEFAULT_CONSISTENCY_PENALTY,
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
        fitness_stddev = statistics.pstdev(
            result.fitness_score for result in agent_results
        )
        risk_adjusted_fitness = average_fitness - (consistency_penalty * fitness_stddev)
        average_wins = sum(
            result.regular_season_wins for result in agent_results
        ) / len(agent_results)
        average_points_for = sum(result.points_for for result in agent_results) / len(agent_results)
        average_playoff_wins = sum(
            result.playoff_wins for result in agent_results
        ) / len(agent_results)
        average_transaction_reward = (
            sum(result.transaction_reward for result in agent_results) / len(agent_results)
        )
        average_playoff_rate = sum(
            result.playoff_rate for result in agent_results
        ) / len(agent_results)
        average_championship_rate = (
            sum(result.championship_rate for result in agent_results) / len(agent_results)
        )
        aggregated_results.append(
            replace(
                agent_results[0],
                fitness_score=round(average_fitness, 2),
                regular_season_wins=round(average_wins, 2),
                points_for=round(average_points_for, 2),
                playoff_seed=None,
                playoff_wins=round(average_playoff_wins, 2),
                champion=average_championship_rate > 0.0,
                transaction_reward=round(average_transaction_reward, 2),
                playoff_rate=round(average_playoff_rate, 4),
                championship_rate=round(average_championship_rate, 4),
                fitness_stddev=round(fitness_stddev, 2),
                risk_adjusted_fitness=round(risk_adjusted_fitness, 2),
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
    consistency_penalty: float = DEFAULT_CONSISTENCY_PENALTY,
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

    return aggregate_scenario_evaluations(
        agents,
        scenario_results,
        consistency_penalty=consistency_penalty,
    )


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
    consistency_penalty: float = DEFAULT_CONSISTENCY_PENALTY,
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
        consistency_penalty=consistency_penalty,
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
    consistency_penalty: float = DEFAULT_CONSISTENCY_PENALTY,
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
            consistency_penalty=consistency_penalty,
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
            evaluated_agent.playoff_rate for evaluated_agent in evaluated_agents
        ) / len(evaluated_agents)
        championship_rate = sum(
            evaluated_agent.championship_rate for evaluated_agent in evaluated_agents
        ) / len(evaluated_agents)
        championship_count = sum(
            evaluated_agent.championship_rate for evaluated_agent in evaluated_agents
        )
        best_evaluated_agent = ranked_agents[0]
        average_fitness_stddev = sum(
            evaluated_agent.fitness_stddev for evaluated_agent in evaluated_agents
        ) / len(evaluated_agents)
        generation_results.append(
            NeuralGenerationResult(
                generation_number=generation_number,
                best_fitness=ranked_agents[0].fitness_score,
                average_fitness=average_fitness,
                average_wins=average_wins,
                average_points_for=average_points_for,
                playoff_rate=playoff_rate,
                championship_count=championship_count,
                championship_rate=championship_rate,
                average_fitness_stddev=average_fitness_stddev,
                best_risk_adjusted_fitness=best_evaluated_agent.risk_adjusted_fitness
                or best_evaluated_agent.fitness_score,
                best_agent_average_wins=best_evaluated_agent.regular_season_wins,
                best_agent_average_points_for=best_evaluated_agent.points_for,
                best_agent_playoff_rate=best_evaluated_agent.playoff_rate,
                best_agent_championship_rate=best_evaluated_agent.championship_rate,
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
                    championship_rate=generation_results[-1].championship_rate,
                    average_fitness_stddev=generation_results[-1].average_fitness_stddev,
                    best_risk_adjusted_fitness=generation_results[-1].best_risk_adjusted_fitness,
                    best_agent_average_wins=generation_results[-1].best_agent_average_wins,
                    best_agent_average_points_for=generation_results[-1].best_agent_average_points_for,
                    best_agent_playoff_rate=generation_results[-1].best_agent_playoff_rate,
                    best_agent_championship_rate=generation_results[-1].best_agent_championship_rate,
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
