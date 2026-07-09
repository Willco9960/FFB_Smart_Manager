from fantasy_engine.leakage_safe_player_pool import (
    build_season_totals,
    create_leakage_safe_player_pool,
)


def create_test_row(
    player_name: str,
    position: str,
    team: str,
    rushing_yards: str = "0",
    rushing_tds: str = "0",
    receiving_yards: str = "0",
    receiving_tds: str = "0",
    receptions: str = "0",
) -> dict[str, str]:
    return {
        "player_name": player_name,
        "position": position,
        "recent_team": team,
        "rushing_yards": rushing_yards,
        "rushing_tds": rushing_tds,
        "receiving_yards": receiving_yards,
        "receiving_tds": receiving_tds,
        "receptions": receptions,
    }


def test_build_season_totals_combines_player_weeks():
    rows = [
        create_test_row(
            player_name="Test RB",
            position="RB",
            team="ATL",
            rushing_yards="100",
        ),
        create_test_row(
            player_name="Test RB",
            position="RB",
            team="ATL",
            rushing_yards="50",
        ),
    ]

    season_totals = build_season_totals(rows)

    player_total = season_totals[("Test RB", "RB")]

    assert player_total["fantasy_points"] == 15.0


def test_create_leakage_safe_player_pool_uses_previous_season_as_projection():
    projection_rows = [
        create_test_row(
            player_name="Test RB",
            position="RB",
            team="ATL",
            rushing_yards="1000",
        ),
    ]
    actual_rows = [
        create_test_row(
            player_name="Test RB",
            position="RB",
            team="ATL",
            rushing_yards="1200",
        ),
    ]

    players = create_leakage_safe_player_pool(
        projection_rows=projection_rows,
        actual_rows=actual_rows,
    )

    assert len(players) == 1
    assert players[0].name == "Test RB"
    assert players[0].projected_score == 100.0
    assert players[0].actual_score == 120.0


def test_create_leakage_safe_player_pool_excludes_players_missing_previous_season():
    projection_rows = []
    actual_rows = [
        create_test_row(
            player_name="Rookie WR",
            position="WR",
            team="NYJ",
            receiving_yards="800",
        ),
    ]

    players = create_leakage_safe_player_pool(
        projection_rows=projection_rows,
        actual_rows=actual_rows,
    )

    assert players == []
