from fantasy_engine.fantasy_points import calculate_fantasy_points
from fantasy_engine.historical_player_pool import create_team_defense_rows
from fantasy_engine.weekly_data import create_weekly_performances


def test_kicker_scoring_uses_field_goal_distance_and_extra_points():
    points = calculate_fantasy_points(
        {
            "position": "K",
            "fg_made_0_19": "1",
            "fg_made_40_49": "1",
            "fg_made_50_59": "1",
            "pat_made": "2",
            "pat_missed": "1",
        }
    )

    assert points == 3.0 + 4.0 + 5.0 + 2.0 - 1.0


def test_defensive_rows_aggregate_into_team_defense():
    rows = [
        {
            "position": "DE",
            "recent_team": "TEST",
            "week": "1",
            "def_sacks": "1",
            "def_interceptions": "0",
            "fumble_recovery_opp": "1",
            "def_tds": "0",
            "def_safeties": "0",
        },
        {
            "position": "CB",
            "recent_team": "TEST",
            "week": "1",
            "def_sacks": "0",
            "def_interceptions": "1",
            "fumble_recovery_opp": "0",
            "def_tds": "1",
            "def_safeties": "0",
        },
    ]

    defense_rows = create_team_defense_rows(rows)
    performances = create_weekly_performances(rows, include_special_teams=True)

    assert len(defense_rows) == 1
    assert defense_rows[0]["player_name"] == "TEST D/ST"
    assert performances[0].position == "DST"
    assert performances[0].fantasy_points == 11.0
