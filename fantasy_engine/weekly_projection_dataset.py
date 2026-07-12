from dataclasses import dataclass

from fantasy_engine.fantasy_points import (
    STANDARD_SCORING,
    FantasyScoringSettings,
    calculate_fantasy_points,
)
from fantasy_engine.historical_loader import RAW_DATA_DIR, load_player_stats
from fantasy_engine.historical_player_pool import DEFENSIVE_POSITIONS
from fantasy_engine.leakage_safe_player_pool import build_season_totals

WEEKLY_PROJECTION_FEATURE_NAMES = (
    "previous_season_points",
    "two_year_weighted_points",
    "recent_3_week_points",
    "recent_3_week_targets",
    "recent_3_week_carries",
    "recent_3_week_receptions",
    "recent_3_week_yards",
    "recent_3_week_touchdowns",
    "games_played_before_week",
    "team_recent_points",
    "opponent_recent_sacks",
    "opponent_recent_interceptions",
    "opponent_recent_fumble_recoveries",
    "opponent_recent_touchdowns_allowed_signal",
    "team_changed",
    "is_qb",
    "is_rb",
    "is_wr",
    "is_te",
    "is_k",
)
ELIGIBLE_POSITIONS = {"QB", "RB", "WR", "TE", "K"}


@dataclass(frozen=True)
class WeeklyProjectionExample:
    player_name: str
    position: str
    season: int
    week: int
    features: tuple[float, ...]
    target_points: float


def get_week(row: dict[str, str]) -> int:
    return int(row.get("week", "0") or "0")


def get_player_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("player_name", ""), row.get("position", ""))


def get_team(row: dict[str, str]) -> str:
    return row.get("recent_team", row.get("team", ""))


def get_float(row: dict[str, str], field_name: str) -> float:
    return float(row.get(field_name, "0") or "0")


def get_row_yards(row: dict[str, str]) -> float:
    return sum(
        get_float(row, field_name)
        for field_name in ("passing_yards", "rushing_yards", "receiving_yards")
    )


def get_row_touchdowns(row: dict[str, str]) -> float:
    return sum(
        get_float(row, field_name)
        for field_name in ("passing_tds", "rushing_tds", "receiving_tds")
    )


def get_row_targets(row: dict[str, str]) -> float:
    return get_float(row, "targets")


def get_row_carries(row: dict[str, str]) -> float:
    return get_float(row, "carries")


def get_row_receptions(row: dict[str, str]) -> float:
    return get_float(row, "receptions")


def get_recent_rows(
    rows: list[dict[str, str]],
    player_key: tuple[str, str],
    week: int,
) -> list[dict[str, str]]:
    prior_rows = [
        row
        for row in rows
        if get_player_key(row) == player_key and get_week(row) < week
    ]
    return sorted(prior_rows, key=get_week)[-3:]


def build_player_week_index(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str], list[dict[str, str]]] = {}

    for row in rows:
        index.setdefault(get_player_key(row), []).append(row)

    for player_rows in index.values():
        player_rows.sort(key=get_week)

    return index


def get_recent_rows_from_index(
    player_week_index: dict[tuple[str, str], list[dict[str, str]]],
    player_key: tuple[str, str],
    week: int,
) -> list[dict[str, str]]:
    prior_rows = [
        row
        for row in player_week_index.get(player_key, [])
        if get_week(row) < week
    ]
    return prior_rows[-3:]


def create_defensive_weekly_signals(
    rows: list[dict[str, str]],
) -> dict[tuple[str, int], dict[str, float]]:
    signals: dict[tuple[str, int], dict[str, float]] = {}
    stat_names = {
        "sacks": "def_sacks",
        "interceptions": "def_interceptions",
        "fumble_recoveries": "fumble_recovery_opp",
        "touchdowns": "def_tds",
    }

    for row in rows:
        if row.get("position", "") not in DEFENSIVE_POSITIONS:
            continue

        key = (get_team(row), get_week(row))
        signals.setdefault(key, {signal_name: 0.0 for signal_name in stat_names})

        for signal_name, source_name in stat_names.items():
            signals[key][signal_name] += get_float(row, source_name)

    return signals


def get_opponent_signal_averages(
    defensive_signals: dict[tuple[str, int], dict[str, float]],
    opponent: str,
    week: int,
) -> tuple[float, float, float, float]:
    prior_signals = [
        signal
        for (team, signal_week), signal in defensive_signals.items()
        if team == opponent and signal_week < week
    ]

    if not prior_signals:
        return (0.0, 0.0, 0.0, 0.0)

    return tuple(
        sum(signal[name] for signal in prior_signals) / len(prior_signals)
        for name in ("sacks", "interceptions", "fumble_recoveries", "touchdowns")
    )


def get_team_recent_points(
    rows: list[dict[str, str]],
    team: str,
    week: int,
) -> float:
    weekly_points: dict[int, float] = {}

    for row in rows:
        row_week = get_week(row)

        if get_team(row) == team and row_week < week and row.get("position") in ELIGIBLE_POSITIONS:
            weekly_points[row_week] = weekly_points.get(row_week, 0.0) + calculate_fantasy_points(
                row,
                STANDARD_SCORING,
            )

    recent_points = sorted(weekly_points.items())[-3:]

    if not recent_points:
        return 0.0

    return sum(points for _, points in recent_points) / len(recent_points)


def build_team_week_points(
    rows: list[dict[str, str]],
    scoring_settings: FantasyScoringSettings,
) -> dict[str, dict[int, float]]:
    team_week_points: dict[str, dict[int, float]] = {}

    for row in rows:
        if row.get("position") not in ELIGIBLE_POSITIONS:
            continue

        team = get_team(row)
        week = get_week(row)
        team_week_points.setdefault(team, {})
        team_week_points[team][week] = team_week_points[team].get(week, 0.0) + (
            calculate_fantasy_points(row, scoring_settings)
        )

    return team_week_points


def get_team_recent_points_from_index(
    team_week_points: dict[str, dict[int, float]],
    team: str,
    week: int,
) -> float:
    prior_points = [
        points
        for row_week, points in sorted(team_week_points.get(team, {}).items())
        if row_week < week
    ]
    recent_points = prior_points[-3:]

    if not recent_points:
        return 0.0

    return sum(recent_points) / len(recent_points)


def create_weekly_projection_examples(
    season_rows_by_season: dict[int, list[dict[str, str]]],
    target_seasons: list[int],
    scoring_settings: FantasyScoringSettings = STANDARD_SCORING,
) -> list[WeeklyProjectionExample]:
    examples = []

    for target_season in target_seasons:
        rows = season_rows_by_season.get(target_season, [])
        prior_rows = season_rows_by_season.get(target_season - 1, [])
        older_rows = season_rows_by_season.get(target_season - 2, [])
        prior_totals = build_season_totals(
            prior_rows,
            scoring_settings,
            include_special_teams=True,
        )
        older_totals = build_season_totals(
            older_rows,
            scoring_settings,
            include_special_teams=True,
        )
        defensive_signals = create_defensive_weekly_signals(rows)
        player_week_index = build_player_week_index(rows)
        team_week_points = build_team_week_points(rows, scoring_settings)

        for row in rows:
            position = row.get("position", "")

            if position not in ELIGIBLE_POSITIONS or not row.get("week"):
                continue

            player_key = get_player_key(row)
            prior_player = prior_totals.get(player_key)

            if prior_player is None:
                continue

            previous_points = float(prior_player["fantasy_points"])
            older_player = older_totals.get(player_key)
            older_points = previous_points

            if older_player is not None:
                older_points = float(older_player["fantasy_points"])

            week = get_week(row)
            recent_rows = get_recent_rows_from_index(player_week_index, player_key, week)
            recent_points = [
                calculate_fantasy_points(recent_row, scoring_settings)
                for recent_row in recent_rows
            ]
            recent_average = sum(recent_points) / len(recent_points) if recent_points else 0.0
            opponent_signals = get_opponent_signal_averages(
                defensive_signals,
                row.get("opponent_team", ""),
                week,
            )
            target_team = get_team(row)
            prior_team = str(prior_player["team"])

            examples.append(
                WeeklyProjectionExample(
                    player_name=row.get("player_name", ""),
                    position=position,
                    season=target_season,
                    week=week,
                    features=(
                        previous_points,
                        (previous_points * 0.65) + (older_points * 0.35),
                        recent_average,
                        sum(get_row_targets(recent_row) for recent_row in recent_rows),
                        sum(get_row_carries(recent_row) for recent_row in recent_rows),
                        sum(get_row_receptions(recent_row) for recent_row in recent_rows),
                        sum(get_row_yards(recent_row) for recent_row in recent_rows),
                        sum(get_row_touchdowns(recent_row) for recent_row in recent_rows),
                        float(len({get_week(recent_row) for recent_row in recent_rows})),
                        get_team_recent_points_from_index(team_week_points, target_team, week),
                        *opponent_signals,
                        float(target_team != prior_team),
                        float(position == "QB"),
                        float(position == "RB"),
                        float(position == "WR"),
                        float(position == "TE"),
                        float(position == "K"),
                    ),
                    target_points=calculate_fantasy_points(row, scoring_settings),
                )
            )

    return examples


def get_weekly_projection_feature_names() -> tuple[str, ...]:
    return WEEKLY_PROJECTION_FEATURE_NAMES


def load_weekly_projection_examples(
    target_seasons: list[int],
    raw_data_dir=RAW_DATA_DIR,
    scoring_settings: FantasyScoringSettings = STANDARD_SCORING,
) -> list[WeeklyProjectionExample]:
    required_seasons = range(min(target_seasons) - 2, max(target_seasons) + 1)
    rows_by_season = {
        season: load_player_stats(season=season, raw_data_dir=raw_data_dir)
        for season in required_seasons
    }

    return create_weekly_projection_examples(
        rows_by_season,
        target_seasons,
        scoring_settings,
    )
