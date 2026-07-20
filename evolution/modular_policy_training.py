"""Evolutionary self-play trainer for the modular manager policy.

Evolution is retained for population diversity and opponent adaptation, but it
mutates the modular policy after projection/behavioral pretraining rather than
trying to learn player projections from championship rewards alone.
"""

import copy
import random

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
    for generation in range(generations):
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
        for scenario_index, (league, performances) in enumerate(scenarios):
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
        history.append(round(averaged[0][0], 2))
        if averaged[0][0] > best_score:
            best_score = averaged[0][0]
            best_policy = clone_modular_policy(averaged[0][1].policy_network)
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
