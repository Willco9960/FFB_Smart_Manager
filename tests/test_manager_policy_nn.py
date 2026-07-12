from pathlib import Path

from agents.neural_draft_agent import NeuralDraftAgent
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from models.manager_policy_nn import (
    MANAGER_FEATURE_COUNT,
    ManagerPolicyNetwork,
    create_draft_action_features,
    load_manager_policy_network,
    save_manager_policy_network,
)


def create_player(name: str, position: str, projected_score: float) -> Player:
    return Player(name=name, position=position, team="TEST", projected_score=projected_score)


def test_draft_action_features_have_stable_feature_count():
    team = Team(name="Team", roster=[create_player("QB", "QB", 20.0)])
    players = [
        create_player("RB One", "RB", 18.0),
        create_player("WR One", "WR", 17.0),
    ]

    features = create_draft_action_features(players[0], team, players)

    assert len(features.values) == MANAGER_FEATURE_COUNT


def test_neural_draft_agent_returns_available_player():
    team = Team(name="Team")
    players = [
        create_player("RB One", "RB", 18.0),
        create_player("WR One", "WR", 17.0),
    ]
    agent = NeuralDraftAgent(policy_network=ManagerPolicyNetwork())

    selected_player = agent.choose_player(players, team, League(name="Test", teams=[team]))

    assert selected_player in players


def test_manager_policy_network_can_be_saved_and_loaded(tmp_path: Path):
    model = ManagerPolicyNetwork()
    output_path = tmp_path / "manager_policy.pt"

    save_manager_policy_network(model, output_path)
    loaded_model = load_manager_policy_network(output_path)

    assert isinstance(loaded_model, ManagerPolicyNetwork)
