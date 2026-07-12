from dataclasses import dataclass

from fantasy_engine.fantasy_points import (
    STANDARD_SCORING,
    FantasyScoringSettings,
    calculate_fantasy_points,
)
from fantasy_engine.historical_loader import RAW_DATA_DIR, load_player_stats
from fantasy_engine.historical_player_pool import (
    FANTASY_RELEVANT_POSITIONS,
    SPECIAL_TEAMS_POSITIONS,
    create_team_defense_rows,
)


@dataclass(frozen=True)
class WeeklyPlayerPerformance:
    player_id: str
    player_name: str
    position: str
    team: str
    week: int
    fantasy_points: float


def get_player_identifier(row: dict[str, str]) -> str:
    player_id = row.get("player_id", "")

    if player_id:
        return player_id

    return f"{row.get('player_name', '')}:{row.get('position', '')}"


def create_weekly_performances(
    rows: list[dict[str, str]],
    scoring_settings: FantasyScoringSettings = STANDARD_SCORING,
    include_special_teams: bool = False,
) -> list[WeeklyPlayerPerformance]:
    performances = []

    source_rows = list(rows)

    if include_special_teams:
        source_rows.extend(create_team_defense_rows(rows))

    for row in source_rows:
        position = row.get("position", "")
        week_text = row.get("week", "")

        valid_positions = FANTASY_RELEVANT_POSITIONS

        if include_special_teams:
            valid_positions = valid_positions | SPECIAL_TEAMS_POSITIONS | {"DST"}

        if position not in valid_positions and position not in {"DST"} or not week_text:
            continue

        performances.append(
            WeeklyPlayerPerformance(
                player_id=get_player_identifier(row),
                player_name=row.get("player_name", ""),
                position=position,
                team=row.get("recent_team", row.get("team", "")),
                week=int(week_text),
                fantasy_points=calculate_fantasy_points(row, scoring_settings),
            )
        )

    return sorted(performances, key=lambda performance: (performance.week, performance.player_id))


def group_performances_by_week(
    performances: list[WeeklyPlayerPerformance],
) -> dict[int, list[WeeklyPlayerPerformance]]:
    performances_by_week: dict[int, list[WeeklyPlayerPerformance]] = {}

    for performance in performances:
        performances_by_week.setdefault(performance.week, []).append(performance)

    return performances_by_week


def get_player_history_before_week(
    performances: list[WeeklyPlayerPerformance],
    player_id: str,
    week: int,
) -> list[WeeklyPlayerPerformance]:
    return [
        performance
        for performance in performances
        if performance.player_id == player_id and performance.week < week
    ]


def load_weekly_performances(
    season: int,
    raw_data_dir=RAW_DATA_DIR,
    scoring_settings: FantasyScoringSettings = STANDARD_SCORING,
    include_special_teams: bool = False,
) -> list[WeeklyPlayerPerformance]:
    rows = load_player_stats(season=season, raw_data_dir=raw_data_dir)

    return create_weekly_performances(
        rows,
        scoring_settings,
        include_special_teams,
    )
