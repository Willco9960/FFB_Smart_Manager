import pytest

from fantasy_engine.historical_seasons import (
    EARLIEST_RELIABLE_SEASON,
    get_required_seasons,
    get_training_seasons,
)


def test_training_seasons_are_inclusive_and_ordered():
    assert get_training_seasons(2020, 2022) == (2020, 2021, 2022)


def test_required_seasons_include_lookback():
    assert get_required_seasons((2020, 2021, 2022), lookback_seasons=2) == (
        2018,
        2019,
        2020,
        2021,
        2022,
    )


def test_training_seasons_reject_future_or_too_old_windows():
    with pytest.raises(ValueError):
        get_training_seasons(EARLIEST_RELIABLE_SEASON - 1, 2020)

    with pytest.raises(ValueError):
        get_training_seasons(2022, 2021)

    with pytest.raises(ValueError):
        get_training_seasons(2020, 2025)
