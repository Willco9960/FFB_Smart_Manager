from fantasy_engine.weekly_data import (
    create_weekly_performances,
    get_player_history_before_week,
    group_performances_by_week,
)


def create_row(player_id: str, player_name: str, position: str, week: str) -> dict[str, str]:
    return {
        "player_id": player_id,
        "player_name": player_name,
        "position": position,
        "recent_team": "ATL",
        "week": week,
        "rushing_yards": "100",
    }


def test_create_weekly_performances_calculates_weekly_points():
    performances = create_weekly_performances([create_row("player-1", "Test RB", "RB", "1")])

    assert len(performances) == 1
    assert performances[0].week == 1
    assert performances[0].fantasy_points == 10.0


def test_create_weekly_performances_filters_non_fantasy_positions():
    performances = create_weekly_performances([create_row("player-1", "Test OL", "OL", "1")])

    assert performances == []


def test_group_performances_by_week_groups_all_players():
    performances = create_weekly_performances(
        [
            create_row("player-1", "Test RB", "RB", "1"),
            create_row("player-2", "Test WR", "WR", "1"),
            create_row("player-1", "Test RB", "RB", "2"),
        ]
    )

    grouped_performances = group_performances_by_week(performances)

    assert len(grouped_performances[1]) == 2
    assert len(grouped_performances[2]) == 1


def test_player_history_excludes_current_and_future_weeks():
    performances = create_weekly_performances(
        [
            create_row("player-1", "Test RB", "RB", "1"),
            create_row("player-1", "Test RB", "RB", "2"),
            create_row("player-1", "Test RB", "RB", "3"),
        ]
    )

    history = get_player_history_before_week(performances, "player-1", week=3)

    assert [performance.week for performance in history] == [1, 2]
