"""Shared historical-season configuration and validation helpers."""

EARLIEST_RELIABLE_SEASON = 2001
LATEST_COMPLETED_SEASON = 2025


def get_training_seasons(
    start_season: int = EARLIEST_RELIABLE_SEASON,
    end_season: int = LATEST_COMPLETED_SEASON - 1,
) -> tuple[int, ...]:
    """Return an inclusive, validated walk-forward training window."""
    if start_season < EARLIEST_RELIABLE_SEASON:
        raise ValueError(f"Training cannot start before {EARLIEST_RELIABLE_SEASON}.")

    if end_season < start_season:
        raise ValueError("Training end season must be >= the start season.")

    if end_season >= LATEST_COMPLETED_SEASON:
        raise ValueError(f"Training must stop before holdout season {LATEST_COMPLETED_SEASON}.")

    return tuple(range(start_season, end_season + 1))


def get_required_seasons(
    training_seasons: tuple[int, ...],
    lookback_seasons: int = 1,
) -> tuple[int, ...]:
    """Return training seasons plus the historical lookback needed by features."""
    if not training_seasons:
        raise ValueError("At least one training season is required.")

    if lookback_seasons < 0:
        raise ValueError("Lookback seasons cannot be negative.")

    first_season = min(training_seasons) - lookback_seasons
    last_season = max(training_seasons)

    return tuple(range(first_season, last_season + 1))
