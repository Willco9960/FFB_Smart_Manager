import pytest

from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from fantasy_engine.transactions import TradeProposal, apply_trade


def create_player(name: str) -> Player:
    return Player(name=name, position="RB", team="TEST")


def create_league() -> tuple[League, Player, Player]:
    first_player = create_player("First Player")
    second_player = create_player("Second Player")
    league = League(
        name="Test League",
        teams=[
            Team(name="Team One", roster=[first_player]),
            Team(name="Team Two", roster=[second_player]),
        ],
    )

    return league, first_player, second_player


def test_apply_trade_swaps_players_between_teams():
    league, first_player, second_player = create_league()
    proposal = TradeProposal(
        proposing_team_name="Team One",
        receiving_team_name="Team Two",
        offered_players=(first_player,),
        requested_players=(second_player,),
        week=4,
    )

    transaction = apply_trade(league, proposal)

    assert second_player in league.teams[0].roster
    assert first_player in league.teams[1].roster
    assert transaction.transaction_type == "trade"


def test_trade_cannot_include_player_not_owned_by_proposing_team():
    league, first_player, second_player = create_league()
    proposal = TradeProposal(
        proposing_team_name="Team One",
        receiving_team_name="Team Two",
        offered_players=(second_player,),
        requested_players=(first_player,),
        week=4,
    )

    with pytest.raises(ValueError, match="does not own"):
        apply_trade(league, proposal)
