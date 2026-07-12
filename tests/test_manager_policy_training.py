from agents.genome_draft_agent import GenomeDraftAgent
from evolution.genome import create_random_genome
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from scripts.train_manager_policy_nn import collect_imitation_examples, train_manager_policy


def test_collect_imitation_examples_returns_action_states():
    league = League(
        name="Training",
        teams=[Team(name="Team")],
        available_players=[
            Player("QB", "QB", "TEST", projected_score=20.0),
            Player("RB", "RB", "TEST", projected_score=18.0),
        ],
    )
    teacher = GenomeDraftAgent(genome=create_random_genome(seed=1))

    examples = collect_imitation_examples(league, teacher, episodes=1, rounds=1)

    assert examples
    assert len(examples[0].features) == 14


def test_train_manager_policy_returns_trained_network():
    examples = [
        type("Example", (), {"features": tuple([0.1] * 14), "target_score": 0.5})()
        for _ in range(4)
    ]

    model, loss = train_manager_policy(examples, epochs=2)

    assert model is not None
    assert loss >= 0.0
