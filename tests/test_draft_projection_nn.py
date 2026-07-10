from fantasy_engine.projection_dataset import SeasonProjectionExample
from models.draft_projection_nn import (
    calculate_mean_absolute_error,
    fit_feature_scaler,
    predict_points,
    train_projection_network,
    transform_features,
)


def create_examples(start: int, count: int) -> list[SeasonProjectionExample]:
    examples = []

    for number in range(start, start + count):
        prior_points = float(number * 10)
        examples.append(
            SeasonProjectionExample(
                player_name=f"Player {number}",
                position="RB",
                season=2021,
                features=(prior_points, prior_points, 0.0, 0.0, 1.0, 0.0, 0.0),
                target_points=prior_points + 10.0,
            )
        )

    return examples


def test_feature_scaler_matches_feature_count():
    scaler = fit_feature_scaler(create_examples(1, 4))

    assert len(scaler.means) == 7
    assert len(scaler.standard_deviations) == 7


def test_transform_features_returns_expected_shape():
    examples = create_examples(1, 3)
    scaler = fit_feature_scaler(examples)

    transformed = transform_features(examples, scaler)

    assert tuple(transformed.shape) == (3, 7)


def test_projection_network_trains_and_predicts():
    training_examples = create_examples(1, 10)
    validation_examples = create_examples(11, 4)

    result = train_projection_network(
        training_examples,
        validation_examples,
        epochs=20,
        patience=10,
    )
    predictions = predict_points(result, validation_examples)

    assert len(predictions) == len(validation_examples)
    assert result.epochs_trained > 0
    assert calculate_mean_absolute_error(predictions, validation_examples) >= 0.0
