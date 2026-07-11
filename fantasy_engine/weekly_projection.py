from dataclasses import replace

from fantasy_engine.player import Player
from fantasy_engine.weekly_data import WeeklyPlayerPerformance

DEFAULT_SEASON_LENGTH = 14
RECENT_HISTORY_WEIGHT = 0.7


def get_player_history_before_week(
    performances: list[WeeklyPlayerPerformance],
    player: Player,
    week: int,
) -> list[WeeklyPlayerPerformance]:
    return [
        performance
        for performance in performances
        if performance.player_name == player.name
        and performance.position == player.position
        and performance.week < week
    ]


def calculate_weekly_projection(
    player: Player,
    history: list[WeeklyPlayerPerformance],
    season_length: int = DEFAULT_SEASON_LENGTH,
) -> float:
    preseason_weekly_projection = player.projected_score / season_length

    if not history:
        return round(preseason_weekly_projection, 2)

    recent_history = history[-3:]
    recent_average = sum(performance.fantasy_points for performance in recent_history) / len(
        recent_history
    )
    projected_points = (recent_average * RECENT_HISTORY_WEIGHT) + (
        preseason_weekly_projection * (1 - RECENT_HISTORY_WEIGHT)
    )

    return round(projected_points, 2)


def create_weekly_projected_roster(
    roster: list[Player],
    performances: list[WeeklyPlayerPerformance],
    week: int,
) -> list[Player]:
    return [
        replace(
            player,
            projected_score=calculate_weekly_projection(
                player,
                get_player_history_before_week(performances, player, week),
            ),
        )
        for player in roster
    ]
