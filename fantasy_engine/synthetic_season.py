import random
from dataclasses import dataclass, replace

from fantasy_engine.weekly_data import WeeklyPlayerPerformance


@dataclass(frozen=True)
class SyntheticSeasonConfig:
    player_role_standard_deviation: float = 0.10
    team_week_standard_deviation: float = 0.08
    weekly_standard_deviation: float = 0.15
    injury_probability: float = 0.015
    maximum_injury_weeks: int = 3


DEFAULT_SYNTHETIC_SEASON_CONFIG = SyntheticSeasonConfig()


def validate_synthetic_config(config: SyntheticSeasonConfig) -> None:
    if config.maximum_injury_weeks < 1:
        raise ValueError("maximum_injury_weeks must be at least 1.")

    for field_name in (
        "player_role_standard_deviation",
        "team_week_standard_deviation",
        "weekly_standard_deviation",
    ):
        if getattr(config, field_name) < 0:
            raise ValueError(f"{field_name} cannot be negative.")

    if not 0 <= config.injury_probability <= 1:
        raise ValueError("injury_probability must be between 0 and 1.")


def generate_synthetic_weekly_performances(
    performances: list[WeeklyPlayerPerformance],
    seed: int = 1,
    config: SyntheticSeasonConfig = DEFAULT_SYNTHETIC_SEASON_CONFIG,
) -> list[WeeklyPlayerPerformance]:
    validate_synthetic_config(config)
    rng = random.Random(seed)
    player_role_multipliers: dict[str, float] = {}
    team_week_multipliers: dict[tuple[str, int], float] = {}
    injury_remaining: dict[str, int] = {}
    synthetic_performances = []

    for performance in sorted(performances, key=lambda item: (item.week, item.player_id)):
        player_multiplier = player_role_multipliers.setdefault(
            performance.player_id,
            rng.lognormvariate(0.0, config.player_role_standard_deviation),
        )
        team_week_multiplier = team_week_multipliers.setdefault(
            (performance.team, performance.week),
            rng.lognormvariate(0.0, config.team_week_standard_deviation),
        )
        remaining = injury_remaining.get(performance.player_id, 0)

        if remaining > 0:
            injury_remaining[performance.player_id] = remaining - 1
            synthetic_points = 0.0
        elif rng.random() < config.injury_probability:
            injury_length = rng.randint(1, config.maximum_injury_weeks)
            injury_remaining[performance.player_id] = injury_length - 1
            synthetic_points = 0.0
        else:
            weekly_multiplier = rng.lognormvariate(0.0, config.weekly_standard_deviation)
            synthetic_points = max(
                0.0,
                performance.fantasy_points
                * player_multiplier
                * team_week_multiplier
                * weekly_multiplier,
            )

        synthetic_performances.append(
            replace(
                performance,
                fantasy_points=round(synthetic_points, 2),
            )
        )

    return synthetic_performances


def generate_synthetic_seasons(
    performances: list[WeeklyPlayerPerformance],
    season_count: int,
    seed: int = 1,
    config: SyntheticSeasonConfig = DEFAULT_SYNTHETIC_SEASON_CONFIG,
) -> list[list[WeeklyPlayerPerformance]]:
    if season_count < 0:
        raise ValueError("season_count cannot be negative.")

    return [
        generate_synthetic_weekly_performances(
            performances=performances,
            seed=seed + season_index,
            config=config,
        )
        for season_index in range(season_count)
    ]
