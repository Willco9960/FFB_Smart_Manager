from pathlib import Path

import torch

from fantasy_engine.projection_dataset import SeasonProjectionExample
from fantasy_engine.weekly_projection_dataset import (
    WEEKLY_PROJECTION_FEATURE_NAMES,
    WeeklyProjectionExample,
)
from models.draft_projection_nn import ProjectionTrainingResult, train_projection_network

DEFAULT_WEEKLY_MODEL_PATH = Path("data/models/weekly_projection_network.pt")


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


def save_weekly_projection_network(
    training_result: ProjectionTrainingResult,
    output_path: Path = DEFAULT_WEEKLY_MODEL_PATH,
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
        },
        output_path,
    )

    return output_path
