from agents.neural_trade_agent import NeuralTradeAgent
from agents.neural_waiver_agent import NeuralWaiverAgent
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from models.manager_policy_nn import ManagerPolicyNetwork


def player(name: str, position: str, projection: float) -> Player:
    return Player(name=name, position=position, team="TEST", projected_score=projection)


def test_neural_waiver_agent_returns_none_when_no_legal_upgrade_exists():
    team = Team(
        name="Team",
        roster=[
            player("QB", "QB", 20),
            player("RB1", "RB", 18),
            player("RB2", "RB", 17),
            player("WR1", "WR", 16),
            player("WR2", "WR", 15),
            player("TE", "TE", 12),
        ],
    )
    league = League(name="Test", teams=[team])

    claim = NeuralWaiverAgent(ManagerPolicyNetwork()).choose_waiver_claim(
        team,
        [player("Weak", "WR", 1)],
        league,
        2,
    )

    assert claim is None


def test_neural_trade_agent_handles_teams_without_a_legal_trade():
    team_one = Team(name="One", roster=[])
    team_two = Team(name="Two", roster=[])
    league = League(name="Test", teams=[team_one, team_two])

    proposal = NeuralTradeAgent(ManagerPolicyNetwork()).choose_trade_proposal(
        team_one,
        [team_two],
        league,
        2,
    )

    assert proposal is None
