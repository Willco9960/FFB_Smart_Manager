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


def test_aggregate_scenario_evaluations_averages_outcome_metrics():
    genome = create_random_genome(seed=1)
    agent = NeuralDraftAgent(ManagerPolicyNetwork(), genome=genome)

    first_result = EvaluatedAgent(
        genome,
        agent,
        100.0,
        regular_season_wins=4,
        points_for=800.0,
        playoff_rate=0.0,
        championship_rate=0.0,
    )
    second_result = EvaluatedAgent(
        genome,
        agent,
        300.0,
        regular_season_wins=10,
        points_for=1200.0,
        playoff_rate=1.0,
        championship_rate=1.0,
    )

    aggregated = aggregate_scenario_evaluations(
        [agent],
        [[first_result], [second_result]],
    )

    assert aggregated[0].regular_season_wins == 7.0
    assert aggregated[0].points_for == 1000.0
    assert aggregated[0].playoff_rate == 0.5
    assert aggregated[0].championship_rate == 0.5
