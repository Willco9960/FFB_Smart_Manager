import copy
import random
from dataclasses import dataclass

import torch

from agents.neural_draft_agent import NeuralDraftAgent
from evolution.full_season import evaluate_full_season_battle_royale
from evolution.genome import DraftStrategyGenome
from evolution.population import EvaluatedAgent, rank_evaluated_agents
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from models.manager_policy_nn import ManagerPolicyNetwork


@dataclass
class NeuralGenerationResult:
    generation_number: int
    best_fitness: float
    average_fitness: float
    best_agent: NeuralDraftAgent
    evaluated_agents: list[EvaluatedAgent]


@dataclass
class NeuralPolicyTrainingResult:
    generations: list[NeuralGenerationResult]
    best_agent: NeuralDraftAgent


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
) -> NeuralPolicyTrainingResult:
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

    for generation_number in range(1, generation_count + 1):
        evaluated_agents = evaluate_full_season_battle_royale(
            agents=agents,
            league=league,
            performances=performances,
            rounds=rounds,
            lineup_rules=lineup_rules,
            seed=seed + generation_number,
            transaction_genome_fallback=transaction_genome,
        )
        ranked_agents = rank_evaluated_agents(evaluated_agents)
        ranked_neural_agents = []

        for evaluated_agent in ranked_agents:
            ranked_neural_agents.append(evaluated_agent.agent)

        selected_agents = ranked_neural_agents[:selection_count]
        generation_results.append(
            NeuralGenerationResult(
                generation_number=generation_number,
                best_fitness=ranked_agents[0].fitness_score,
                average_fitness=sum(
                    evaluated_agent.fitness_score for evaluated_agent in evaluated_agents
                )
                / len(evaluated_agents),
                best_agent=selected_agents[0],
                evaluated_agents=evaluated_agents,
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
