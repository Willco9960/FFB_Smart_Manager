from agents.neural_draft_agent import NeuralDraftAgent
from evolution.genome import create_random_genome
from evolution.neural_policy_training import aggregate_scenario_evaluations
from evolution.population import EvaluatedAgent
from models.manager_policy_nn import ManagerPolicyNetwork


def test_aggregate_scenario_evaluations_averages_each_agent():
    genome = create_random_genome(seed=1)
    first_agent = NeuralDraftAgent(ManagerPolicyNetwork(), genome=genome)
    second_agent = NeuralDraftAgent(ManagerPolicyNetwork(), genome=genome)

    first_results = [
        EvaluatedAgent(genome, first_agent, 100.0),
        EvaluatedAgent(genome, second_agent, 200.0),
    ]
    second_results = [
        EvaluatedAgent(genome, first_agent, 300.0),
        EvaluatedAgent(genome, second_agent, 400.0),
    ]

    aggregated = aggregate_scenario_evaluations(
        [first_agent, second_agent],
        [first_results, second_results],
    )

    assert [result.fitness_score for result in aggregated] == [200.0, 300.0]
