from fantasy_engine.weekly_projection_dataset import WeeklyProjectionExample
from models.weekly_projection_nn import (
    calculate_weekly_heuristic_projection,
    convert_weekly_examples,
)


def test_convert_weekly_examples_preserves_features_and_target():
    example = WeeklyProjectionExample(
        player_name="Player",
        position="WR",
        season=2022,
        week=4,
        features=tuple([1.0] * 20),
        target_points=18.0,
    )

    converted = convert_weekly_examples([example])[0]

    assert converted.features == example.features
    assert converted.target_points == example.target_points


def test_weekly_heuristic_projection_uses_preseason_value_before_recent_history():
    example = WeeklyProjectionExample(
        player_name="Player",
        position="RB",
        season=2022,
        week=1,
        features=(210.0, 0.0, 0.0, *([0.0] * 17)),
        target_points=18.0,
    )

    assert calculate_weekly_heuristic_projection(example) == 15.0
