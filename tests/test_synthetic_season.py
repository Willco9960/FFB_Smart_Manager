import pytest

from fantasy_engine.synthetic_season import (
    SyntheticSeasonConfig,
    generate_synthetic_seasons,
    generate_synthetic_weekly_performances,
)
from fantasy_engine.weekly_data import WeeklyPlayerPerformance


def create_performances() -> list[WeeklyPlayerPerformance]:
    return [
        WeeklyPlayerPerformance(
            player_id="player-1",
            player_name="Player One",
            position="WR",
            team="ATL",
            week=week,
            fantasy_points=10.0,
        )
        for week in range(1, 5)
    ]


def test_synthetic_season_is_reproducible_with_same_seed():
    performances = create_performances()

    first = generate_synthetic_weekly_performances(performances, seed=42)
    second = generate_synthetic_weekly_performances(performances, seed=42)

    assert first == second


def test_synthetic_season_preserves_rows_and_nonnegative_points():
    synthetic = generate_synthetic_weekly_performances(
        create_performances(),
        seed=42,
        config=SyntheticSeasonConfig(injury_probability=0.0),
    )

    assert len(synthetic) == 4
    assert all(performance.fantasy_points >= 0 for performance in synthetic)
    assert [performance.week for performance in synthetic] == [1, 2, 3, 4]


def test_synthetic_season_count_and_invalid_count():
    assert len(generate_synthetic_seasons(create_performances(), 3)) == 3

    with pytest.raises(ValueError):
        generate_synthetic_seasons(create_performances(), -1)
