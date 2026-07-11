from fantasy_engine.player import Player
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from fantasy_engine.weekly_projection import (
    calculate_weekly_projection,
    create_weekly_projected_roster,
)


def create_performance(week: int, points: float) -> WeeklyPlayerPerformance:
    return WeeklyPlayerPerformance(
        player_id="player-1",
        player_name="Test RB",
        position="RB",
        team="ATL",
        week=week,
        fantasy_points=points,
    )


def test_week_one_projection_uses_preseason_value():
    player = Player(name="Test RB", position="RB", team="ATL", projected_score=140.0)

    projection = calculate_weekly_projection(player, [])

    assert projection == 10.0


def test_weekly_projection_uses_only_prior_week_history():
    player = Player(name="Test RB", position="RB", team="ATL", projected_score=140.0)
    performances = [create_performance(1, 20.0), create_performance(2, 5.0)]

    projected_roster = create_weekly_projected_roster([player], performances, week=3)

    assert projected_roster[0].projected_score == 11.75


def test_weekly_projection_ignores_current_and_future_week_results():
    player = Player(name="Test RB", position="RB", team="ATL", projected_score=140.0)
    performances = [
        create_performance(1, 20.0),
        create_performance(2, 5.0),
        create_performance(3, 100.0),
    ]

    projected_roster = create_weekly_projected_roster([player], performances, week=3)

    assert projected_roster[0].projected_score == 11.75
