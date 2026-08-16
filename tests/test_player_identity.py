from fantasy_engine.historical_player_pool import get_player_id
from fantasy_engine.leakage_safe_player_pool import create_leakage_safe_player_pool


def test_authoritative_player_id_survives_name_and_team_change():
    first = {
        "player_id": "00-123",
        "player_name": "Old Name",
        "position": "WR",
        "recent_team": "OLD",
    }
    second = {
        "player_id": "00-123",
        "player_name": "New Name",
        "position": "WR",
        "recent_team": "NEW",
    }

    assert get_player_id(first) == get_player_id(second) == "00-123"


def test_projection_only_player_remains_in_pool_with_missing_actual_score():
    projection_rows = [
        {
            "player_id": "rookie-1",
            "player_name": "Rookie WR",
            "position": "WR",
            "recent_team": "NEW",
            "receiving_yards": "100",
        }
    ]

    players = create_leakage_safe_player_pool(
        projection_rows=projection_rows,
        actual_rows=[],
    )

    assert len(players) == 1
    assert players[0].player_id == "rookie-1"
    assert players[0].projected_score == 10.0
    assert players[0].actual_score == 0.0
    assert players[0].history_missing is False


def test_actual_only_player_remains_in_pool_with_missing_projection_history():
    actual_rows = [
        {
            "player_id": "rookie-2",
            "player_name": "Rookie RB",
            "position": "RB",
            "recent_team": "NEW",
            "rushing_yards": "100",
        }
    ]

    players = create_leakage_safe_player_pool(
        projection_rows=[],
        actual_rows=actual_rows,
    )

    assert len(players) == 1
    assert players[0].player_id == "rookie-2"
    assert players[0].projected_score == 0.0
    assert players[0].actual_score == 10.0
    assert players[0].history_missing is True
