from agents.waiver_agent import GenomeWaiverAgent
from evolution.genome import DraftStrategyGenome
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team


def create_player(name: str, position: str, projected_score: float) -> Player:
    return Player(
        name=name,
        position=position,
        team="TEST",
        projected_score=projected_score,
    )


def create_agent() -> GenomeWaiverAgent:
    return GenomeWaiverAgent(
        genome=DraftStrategyGenome(
            projection_weight=1.0,
            position_scarcity_weight=0.0,
            adp_value_weight=0.0,
            upside_weight=0.0,
            floor_weight=0.0,
            bye_week_penalty=0.0,
            qb_priority=0.0,
            rb_priority=0.0,
            wr_priority=0.0,
            te_priority=0.0,
        )
    )


def test_waiver_agent_replaces_a_low_value_extra_player_with_better_free_agent():
    roster = [
        create_player("QB", "QB", 20.0),
        create_player("RB One", "RB", 18.0),
        create_player("RB Two", "RB", 17.0),
        create_player("RB Bench", "RB", 1.0),
        create_player("WR One", "WR", 16.0),
        create_player("WR Two", "WR", 15.0),
        create_player("TE", "TE", 12.0),
    ]
    free_agent = create_player("Free Agent", "WR", 14.0)
    team = Team(name="Team", roster=roster)
    league = League(name="Test", teams=[team], available_players=[free_agent])

    claim = create_agent().choose_waiver_claim(team, [free_agent], league, week=2)

    assert claim is not None
    assert claim.add_player == free_agent
    assert claim.drop_player.name == "RB Bench"


def test_waiver_agent_does_not_drop_a_required_starter_when_roster_is_thin():
    roster = [
        create_player("QB", "QB", 20.0),
        create_player("RB One", "RB", 18.0),
        create_player("RB Two", "RB", 17.0),
        create_player("WR One", "WR", 16.0),
        create_player("WR Two", "WR", 15.0),
        create_player("TE", "TE", 1.0),
    ]
    free_agent = create_player("Free Agent", "WR", 20.0)
    team = Team(name="Team", roster=roster)
    league = League(name="Test", teams=[team], available_players=[free_agent])

    claim = create_agent().choose_waiver_claim(team, [free_agent], league, week=2)

    assert claim is None
