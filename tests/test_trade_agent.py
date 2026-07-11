from agents.trade_agent import GenomeTradeAgent
from evolution.genome import DraftStrategyGenome
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team


def create_player(name: str, position: str, projected_score: float) -> Player:
    return Player(name=name, position=position, team="TEST", projected_score=projected_score)


def create_agent() -> GenomeTradeAgent:
    return GenomeTradeAgent(
        genome=DraftStrategyGenome(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )


def test_trade_agent_proposes_trade_that_improves_both_starting_lineups():
    first_team = Team(
        name="Team One",
        roster=[
            create_player("QB One", "QB", 20.0),
            create_player("RB One", "RB", 18.0),
            create_player("RB Two", "RB", 17.0),
            create_player("RB Bench", "RB", 16.0),
            create_player("WR One", "WR", 15.0),
            create_player("WR Two", "WR", 14.0),
            create_player("TE One", "TE", 20.0),
            create_player("TE Bench", "TE", 13.0),
        ],
    )
    second_team = Team(
        name="Team Two",
        roster=[
            create_player("QB Two", "QB", 20.0),
            create_player("RB Three", "RB", 20.0),
            create_player("RB Four", "RB", 19.0),
            create_player("WR Three", "WR", 21.0),
            create_player("WR Four", "WR", 20.0),
            create_player("WR Bench", "WR", 17.0),
            create_player("TE Two", "TE", 12.0),
            create_player("RB Bench", "RB", 18.0),
        ],
    )
    league = League(name="Test", teams=[first_team, second_team])

    proposal = create_agent().choose_trade_proposal(
        first_team,
        [second_team],
        league,
        week=3,
    )

    assert proposal is not None
    assert proposal.proposing_team_name == "Team One"
    assert proposal.receiving_team_name == "Team Two"
