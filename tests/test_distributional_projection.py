from fantasy_engine.projection_dataset import SeasonProjectionExample
from models.distributional_projection import (
    predict_distributions,
    train_distributional_projection_network,
)


def example(number: int) -> SeasonProjectionExample:
    return SeasonProjectionExample(
        player_name=f"Player {number}",
        position="RB",
        season=2020 + number // 10,
        features=(float(number), float(number), 0.0, 0.0, 1.0, 0.0, 0.0),
        target_points=float(number * 2),
    )


def test_distributional_projection_produces_floor_median_ceiling_and_boom():
    result = train_distributional_projection_network(
        [example(number) for number in range(1, 15)],
        [example(number) for number in range(15, 20)],
        epochs=5,
        patience=5,
    )
    predictions = predict_distributions(result, [example(20)])
    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.floor >= 0.0
    assert prediction.floor <= prediction.ceiling
    assert 0.0 <= prediction.boom_probability <= 1.0
