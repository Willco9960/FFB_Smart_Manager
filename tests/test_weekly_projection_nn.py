from fantasy_engine.weekly_projection_dataset import WeeklyProjectionExample
from models.weekly_projection_nn import convert_weekly_examples


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
