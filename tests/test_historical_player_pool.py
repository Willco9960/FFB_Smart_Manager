from fantasy_engine.draft import run_snake_draft
from fantasy_engine.fantasy_points import HALF_PPR_SCORING
from fantasy_engine.historical_player_pool import (
    create_historical_player_pool,
    create_player_from_historical_row,
    get_player_name,
    get_player_position,
    get_player_team,
    is_fantasy_relevant_row,
)
from fantasy_engine.league import League
from fantasy_engine.team import Team


def test_is_fantasy_relevant_row_returns_true_for_skill_positions():
    assert is_fantasy_relevant_row({"position": "QB"})
    assert is_fantasy_relevant_row({"position": "RB"})
    assert is_fantasy_relevant_row({"position": "WR"})
    assert is_fantasy_relevant_row({"position": "TE"})


def test_is_fantasy_relevant_row_returns_false_for_other_positions():
    assert not is_fantasy_relevant_row({"position": "K"})
    assert not is_fantasy_relevant_row({"position": "OL"})
    assert not is_fantasy_relevant_row({"position": ""})


def test_get_player_fields_from_historical_row():
    row = {
        "player_name": "Test Player",
        "position": "WR",
        "recent_team": "ATL",
    }

    assert get_player_name(row) == "Test Player"
    assert get_player_position(row) == "WR"
    assert get_player_team(row) == "ATL"


def test_get_player_team_falls_back_to_team_column():
    row = {
        "player_name": "Test Player",
        "position": "RB",
        "recent_team": "None",
        "team": "IND",
    }

    assert get_player_team(row) == "IND"


def test_create_player_from_historical_row_uses_actual_fantasy_points():
    row = {
        "player_name": "Test RB",
        "position": "RB",
        "recent_team": "DAL",
        "rushing_yards": "100",
        "rushing_tds": "1",
    }

    player = create_player_from_historical_row(row)

    assert player.name == "Test RB"
    assert player.position == "RB"
    assert player.team == "DAL"
    assert player.projected_score == 0.0
    assert player.actual_score == 16.0


def test_create_player_from_historical_row_supports_half_ppr():
    row = {
        "player_name": "Test WR",
        "position": "WR",
        "recent_team": "MIN",
        "receptions": "4",
        "receiving_yards": "80",
        "receiving_tds": "1",
    }

    player = create_player_from_historical_row(row, HALF_PPR_SCORING)

    assert player.actual_score == 16.0


def test_create_historical_player_pool_filters_to_fantasy_relevant_players():
    rows = [
        {
            "player_name": "Useful QB",
            "position": "QB",
            "recent_team": "BUF",
            "passing_yards": "250",
            "passing_tds": "2",
        },
        {
            "player_name": "Ignored Lineman",
            "position": "OL",
            "recent_team": "BUF",
        },
    ]

    players = create_historical_player_pool(rows)

    assert len(players) == 1
    assert players[0].name == "Useful QB"
    assert players[0].actual_score == 18.0


def test_historical_player_pool_can_be_passed_to_draft_simulator():
    rows = []

    for number in range(1, 21):
        rows.append(
            {
                "player_name": f"Player {number}",
                "position": "RB",
                "recent_team": "ATL",
                "rushing_yards": str(100 - number),
            }
        )

    players = create_historical_player_pool(rows)
    teams = [Team(name=f"Team {number}") for number in range(1, 5)]
    league = League(
        name="Historical Test League",
        teams=teams,
        available_players=players,
    )

    draft_results = run_snake_draft(league, rounds=2)

    assert len(draft_results) == 8
    assert len(league.available_players) == 12

    for team in league.teams:
        assert team.roster_size() == 2
