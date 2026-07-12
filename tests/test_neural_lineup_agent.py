from agents.neural_lineup_agent import NeuralLineupAgent
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES
from fantasy_engine.player import Player
from models.manager_policy_nn import ManagerPolicyNetwork


def test_neural_lineup_agent_returns_a_legal_lineup():
    roster = [
        Player("QB", "QB", "TEST", projected_score=20.0),
        Player("RB1", "RB", "TEST", projected_score=18.0),
        Player("RB2", "RB", "TEST", projected_score=17.0),
        Player("WR1", "WR", "TEST", projected_score=16.0),
        Player("WR2", "WR", "TEST", projected_score=15.0),
        Player("TE", "TE", "TEST", projected_score=12.0),
        Player("Bench", "WR", "TEST", projected_score=1.0),
    ]
    lineup = NeuralLineupAgent(
        policy_network=ManagerPolicyNetwork(),
        lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
    ).choose_lineup(roster)

    assert lineup.is_complete()
    assert len(lineup.players) == 7
