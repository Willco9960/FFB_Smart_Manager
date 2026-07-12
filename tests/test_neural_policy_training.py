from agents.neural_draft_agent import NeuralDraftAgent
from evolution.genome import create_random_genome
from evolution.neural_policy_training import create_next_neural_generation, mutate_policy_network
from models.manager_policy_nn import ManagerPolicyNetwork


def test_mutate_policy_network_preserves_network_interface():
    network = ManagerPolicyNetwork()
    mutated = mutate_policy_network(network, __import__("random").Random(1))

    assert isinstance(mutated, ManagerPolicyNetwork)
    assert mutated is not network


def test_create_next_neural_generation_keeps_selected_agents():
    genome = create_random_genome(seed=1)
    agent = NeuralDraftAgent(ManagerPolicyNetwork(), genome=genome)

    next_generation = create_next_neural_generation(
        selected_agents=[agent],
        population_size=3,
        seed=1,
        mutation_strength=0.01,
    )

    assert len(next_generation) == 3
    assert next_generation[0] is agent
