import pytest

from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.season import TeamStanding
from fantasy_engine.team import Team
from fantasy_engine.transactions import (
    TradeProposal,
    TransactionValueTracker,
    WaiverClaim,
    apply_trade,
    apply_waiver_claim,
    create_inverse_standings_waiver_order,
    process_waiver_claims,
)


def create_player(name: str, position: str = "RB") -> Player:
    return Player(name=name, position=position, team="TEST")


def create_league() -> tuple[League, Player, Player, Player]:
    roster_player_one = create_player("Roster Player One")
    roster_player_two = create_player("Roster Player Two")
    free_agent = create_player("Free Agent")
    league = League(
        name="Test League",
        teams=[
            Team(name="Team One", roster=[roster_player_one]),
            Team(name="Team Two", roster=[roster_player_two]),
        ],
        available_players=[free_agent],
    )

    return league, roster_player_one, roster_player_two, free_agent


def test_inverse_standings_waiver_order_prioritizes_worse_record():
    standings = {
        "Team One": TeamStanding(team_name="Team One", wins=5, losses=1, points_for=600.0),
        "Team Two": TeamStanding(team_name="Team Two", wins=1, losses=5, points_for=400.0),
    }

    assert create_inverse_standings_waiver_order(standings) == ["Team Two", "Team One"]


def test_apply_waiver_claim_swaps_a_free_agent_for_rostered_player():
    league, roster_player_one, _, free_agent = create_league()
    claim = WaiverClaim(
        team_name="Team One",
        add_player=free_agent,
        drop_player=roster_player_one,
        week=2,
    )

    transaction = apply_waiver_claim(league, claim)

    assert free_agent in league.teams[0].roster
    assert roster_player_one in league.available_players
    assert transaction.added_player_name == "Free Agent"


def test_waiver_claim_cannot_add_a_player_already_rostered_by_another_team():
    league, roster_player_one, roster_player_two, _ = create_league()
    claim = WaiverClaim(
        team_name="Team One",
        add_player=roster_player_two,
        drop_player=roster_player_one,
        week=2,
    )

    with pytest.raises(ValueError, match="not a free agent"):
        apply_waiver_claim(league, claim)


def test_waiver_order_awards_contested_free_agent_to_higher_priority_team():
    league, roster_player_one, roster_player_two, free_agent = create_league()
    claims = [
        WaiverClaim("Team One", free_agent, roster_player_one, week=2),
        WaiverClaim("Team Two", free_agent, roster_player_two, week=2),
    ]

    result = process_waiver_claims(
        league=league,
        claims=claims,
        waiver_order=["Team Two", "Team One"],
    )

    assert result.processed_claims == [claims[1]]
    assert result.rejected_claims == [claims[0]]
    assert free_agent in league.teams[1].roster


def test_transaction_value_tracker_rewards_a_successful_waiver_pickup():
    league, roster_player_one, _, free_agent = create_league()
    claim = WaiverClaim("Team One", free_agent, roster_player_one, week=2)
    transaction = apply_waiver_claim(league, claim)
    tracker = TransactionValueTracker()
    tracker.register([transaction])

    impacts = tracker.evaluate_week(
        week=2,
        weekly_points_by_player={
            ("Free Agent", "RB"): 18.0,
            ("Roster Player One", "RB"): 6.0,
        },
    )

    assert impacts[0].net_points == 12.0
    assert impacts[0].reward == 12.0
    assert impacts[0].outcome == "positive"


def test_transaction_value_tracker_evaluates_trade_for_both_teams():
    first_player = create_player("First Player")
    second_player = create_player("Second Player")
    league = League(
        name="Trade League",
        teams=[
            Team(name="Team One", roster=[first_player]),
            Team(name="Team Two", roster=[second_player]),
        ],
    )
    transaction = apply_trade(
        league,
        TradeProposal(
            proposing_team_name="Team One",
            receiving_team_name="Team Two",
            offered_players=(first_player,),
            requested_players=(second_player,),
            week=3,
        ),
    )
    tracker = TransactionValueTracker()
    tracker.register([transaction])

    impacts = tracker.evaluate_week(
        week=3,
        weekly_points_by_player={
            ("First Player", "RB"): 4.0,
            ("Second Player", "RB"): 14.0,
        },
    )

    assert len(impacts) == 2
    assert impacts[0].net_points == 10.0
    assert impacts[1].net_points == -10.0
