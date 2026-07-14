from pathlib import Path

import torch

from fantasy_engine.projection_dataset import SeasonProjectionExample
from fantasy_engine.weekly_projection_dataset import (
    WEEKLY_PROJECTION_FEATURE_NAMES,
    WeeklyProjectionExample,
)
from models.draft_projection_nn import (
    ProjectionTrainingResult,
    predict_points,
    train_projection_network,
)

DEFAULT_WEEKLY_MODEL_PATH = Path("data/models/weekly_projection_network.pt")
DEFAULT_WEEKLY_SEASON_LENGTH = 14


def convert_weekly_examples(
    examples: list[WeeklyProjectionExample],
) -> list[SeasonProjectionExample]:
    return [
        SeasonProjectionExample(
            player_name=example.player_name,
            position=example.position,
            season=example.season,
            features=example.features,
            target_points=example.target_points,
        )
        for example in examples
    ]


def train_weekly_projection_network(
    training_examples: list[WeeklyProjectionExample],
    validation_examples: list[WeeklyProjectionExample],
    **kwargs,
) -> ProjectionTrainingResult:
    return train_projection_network(
        training_examples=convert_weekly_examples(training_examples),
        validation_examples=convert_weekly_examples(validation_examples),
        **kwargs,
    )


def calculate_weekly_heuristic_projection(
    example: WeeklyProjectionExample,
    season_length: int = DEFAULT_WEEKLY_SEASON_LENGTH,
) -> float:
    previous_season_projection = example.features[0] / season_length
    recent_games = example.features[8]

    if recent_games <= 0:
        return previous_season_projection

    recent_average = example.features[2]
    return (recent_average * 0.7) + (previous_season_projection * 0.3)


def calibrate_neural_weights_by_position(
    training_result: ProjectionTrainingResult,
    validation_examples: list[WeeklyProjectionExample],
    candidate_weights: tuple[float, ...] = tuple(index / 20 for index in range(21)),
) -> dict[str, float]:
    converted_examples = convert_weekly_examples(validation_examples)
    neural_predictions = predict_points(training_result, converted_examples)
    grouped_examples: dict[str, list[tuple[WeeklyProjectionExample, float]]] = {}

    for example, prediction in zip(validation_examples, neural_predictions, strict=True):
        grouped_examples.setdefault(example.position, []).append((example, prediction))

    calibrated_weights = {}

    for position, position_examples in grouped_examples.items():
        best_weight = 1.0
        best_error = float("inf")

        for neural_weight in candidate_weights:
            error = sum(
                abs(
                    (
                        neural_weight * neural_prediction
                        + (1 - neural_weight) * calculate_weekly_heuristic_projection(example)
                    )
                    - example.target_points
                )
                for example, neural_prediction in position_examples
            ) / len(position_examples)

            if error < best_error:
                best_error = error
                best_weight = neural_weight

        calibrated_weights[position] = round(best_weight, 2)

    return calibrated_weights


def predict_calibrated_weekly_points(
    training_result: ProjectionTrainingResult,
    examples: list[WeeklyProjectionExample],
    neural_weights_by_position: dict[str, float],
) -> list[float]:
    neural_predictions = predict_points(
        training_result,
        convert_weekly_examples(examples),
    )

    return [
        round(
            (
                neural_weights_by_position.get(example.position, 1.0) * neural_prediction
                + (1 - neural_weights_by_position.get(example.position, 1.0))
                * calculate_weekly_heuristic_projection(example)
            ),
            2,
        )
        for example, neural_prediction in zip(examples, neural_predictions, strict=True)
    ]


def save_weekly_projection_network(
    training_result: ProjectionTrainingResult,
    output_path: Path = DEFAULT_WEEKLY_MODEL_PATH,
    neural_weights_by_position: dict[str, float] | None = None,
    training_seasons: tuple[int, ...] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_size": len(training_result.feature_scaler.means),
            "state_dict": training_result.model.state_dict(),
            "feature_means": training_result.feature_scaler.means,
            "feature_standard_deviations": training_result.feature_scaler.standard_deviations,
            "target_mean": training_result.target_mean,
            "target_standard_deviation": training_result.target_standard_deviation,
            "feature_names": list(WEEKLY_PROJECTION_FEATURE_NAMES),
            "neural_weights_by_position": neural_weights_by_position or {},
            "training_seasons": list(training_seasons or []),
            "max_training_season": max(training_seasons) if training_seasons else None,
        },
        output_path,
    )

    return output_path
