from dataclasses import dataclass

from fantasy_engine.fantasy_points import STANDARD_SCORING, FantasyScoringSettings
from fantasy_engine.historical_loader import RAW_DATA_DIR, load_player_stats
from fantasy_engine.leakage_safe_player_pool import build_season_totals

POSITION_ORDER = ("QB", "RB", "WR", "TE")


@dataclass(frozen=True)
class SeasonProjectionExample:
    player_name: str
    position: str
    season: int
    features: tuple[float, ...]
    target_points: float


def get_feature_names() -> tuple[str, ...]:
    return (
        "previous_season_points",
        "two_year_average_points",
        "team_changed",
        "is_qb",
        "is_rb",
        "is_wr",
        "is_te",
    )


def get_position_features(position: str) -> tuple[float, ...]:
    return tuple(float(position == expected_position) for expected_position in POSITION_ORDER)


def create_projection_examples(
    season_totals_by_season: dict[int, dict[tuple[str, str], dict[str, str | float]]],
    target_seasons: list[int],
) -> list[SeasonProjectionExample]:
    examples = []

    for target_season in target_seasons:
        prior_totals = season_totals_by_season.get(target_season - 1, {})
        older_totals = season_totals_by_season.get(target_season - 2, {})
        target_totals = season_totals_by_season.get(target_season, {})

        for player_key, target_player in target_totals.items():
            prior_player = prior_totals.get(player_key)

            if prior_player is None:
                continue

            prior_points = float(prior_player["fantasy_points"])
            older_player = older_totals.get(player_key)
            older_points = prior_points

            if older_player is not None:
                older_points = float(older_player["fantasy_points"])
            target_team = str(target_player["team"])
            prior_team = str(prior_player["team"])
            position = str(target_player["position"])

            examples.append(
                SeasonProjectionExample(
                    player_name=str(target_player["name"]),
                    position=position,
                    season=target_season,
                    features=(
                        prior_points,
                        (prior_points + older_points) / 2,
                        float(target_team != prior_team),
                        *get_position_features(position),
                    ),
                    target_points=float(target_player["fantasy_points"]),
                )
            )

    return examples


def load_projection_examples(
    target_seasons: list[int],
    raw_data_dir=RAW_DATA_DIR,
    scoring_settings: FantasyScoringSettings = STANDARD_SCORING,
) -> list[SeasonProjectionExample]:
    required_seasons = range(min(target_seasons) - 1, max(target_seasons) + 1)
    season_totals_by_season = {}

    for season in required_seasons:
        rows = load_player_stats(season=season, raw_data_dir=raw_data_dir)
        season_totals_by_season[season] = build_season_totals(rows, scoring_settings)

    return create_projection_examples(season_totals_by_season, target_seasons)
