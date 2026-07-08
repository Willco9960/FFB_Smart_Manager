from fantasy_engine.fantasy_points import (
    HALF_PPR_SCORING,
    PPR_SCORING,
    STANDARD_SCORING,
    calculate_fantasy_points,
    get_stat_value,
)


def test_get_stat_value_returns_zero_for_missing_stat():
    stats = {}

    value = get_stat_value(stats, "passing_yards")

    assert value == 0.0


def test_get_stat_value_converts_string_to_float():
    stats = {"passing_yards": "250"}

    value = get_stat_value(stats, "passing_yards")

    assert value == 250.0


def test_calculate_passing_fantasy_points():
    stats = {
        "passing_yards": "300",
        "passing_tds": "2",
        "interceptions": "1",
    }

    points = calculate_fantasy_points(stats, STANDARD_SCORING)

    assert points == 18.0


def test_calculate_rushing_and_receiving_standard_points():
    stats = {
        "rushing_yards": "50",
        "rushing_tds": "1",
        "receptions": "4",
        "receiving_yards": "70",
        "receiving_tds": "1",
    }

    points = calculate_fantasy_points(stats, STANDARD_SCORING)

    assert points == 24.0


def test_calculate_rushing_and_receiving_half_ppr_points():
    stats = {
        "rushing_yards": "50",
        "rushing_tds": "1",
        "receptions": "4",
        "receiving_yards": "70",
        "receiving_tds": "1",
    }

    points = calculate_fantasy_points(stats, HALF_PPR_SCORING)

    assert points == 26.0


def test_calculate_rushing_and_receiving_ppr_points():
    stats = {
        "rushing_yards": "50",
        "rushing_tds": "1",
        "receptions": "4",
        "receiving_yards": "70",
        "receiving_tds": "1",
    }

    points = calculate_fantasy_points(stats, PPR_SCORING)

    assert points == 28.0


def test_calculate_turnover_penalties():
    stats = {
        "passing_yards": "100",
        "interceptions": "2",
        "fumbles_lost": "1",
    }

    points = calculate_fantasy_points(stats, STANDARD_SCORING)

    assert points == -2.0
