from dataclasses import replace
from pathlib import Path

import torch

from fantasy_engine.historical_loader import RAW_DATA_DIR
from fantasy_engine.player import Player
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from fantasy_engine.weekly_projection import (
    calculate_weekly_projection,
    get_player_history_before_week,
)
from fantasy_engine.weekly_projection_dataset import (
    WEEKLY_PROJECTION_FEATURE_NAMES,
    load_weekly_projection_examples,
)
from models.draft_projection_nn import (
    DraftProjectionNetwork,
    FeatureScaler,
    ProjectionTrainingResult,
    predict_points,
)
from models.weekly_projection_nn import (
    calculate_weekly_heuristic_projection,
    convert_weekly_examples,
)

DEFAULT_WEEKLY_MODEL_PATH = Path("data/models/weekly_projection_network.pt")


class WeeklyNeuralProjectionService:
    def __init__(
        self,
        training_result: ProjectionTrainingResult,
        predictions: dict[tuple[int, str, str], float],
        neural_weights_by_position: dict[str, float] | None = None,
        use_calibrated_weights: bool = False,
    ):
        self.training_result = training_result
        self.predictions = predictions
        self.neural_weights_by_position = (
            neural_weights_by_position or {} if use_calibrated_weights else {}
        )

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Path,
        target_season: int,
        raw_data_dir=RAW_DATA_DIR,
        use_calibrated_weights: bool = False,
    ) -> "WeeklyNeuralProjectionService":
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        model = DraftProjectionNetwork(input_size=checkpoint["input_size"])
        model.load_state_dict(checkpoint["state_dict"])
        feature_scaler = FeatureScaler(
            means=tuple(checkpoint["feature_means"]),
            standard_deviations=tuple(checkpoint["feature_standard_deviations"]),
        )
        training_result = ProjectionTrainingResult(
            model=model,
            feature_scaler=feature_scaler,
            target_mean=float(checkpoint["target_mean"]),
            target_standard_deviation=float(checkpoint["target_standard_deviation"]),
            best_validation_loss=float(checkpoint.get("best_validation_loss", 0.0)),
            epochs_trained=int(checkpoint.get("epochs_trained", 0)),
        )
        weekly_examples = load_weekly_projection_examples(
            [target_season],
            raw_data_dir=raw_data_dir,
        )
        predictions = predict_points(
            training_result,
            convert_weekly_examples(weekly_examples),
        )
        neural_weights_by_position = {
            position: float(weight)
            for position, weight in checkpoint.get("neural_weights_by_position", {}).items()
        }
        applied_weights = neural_weights_by_position if use_calibrated_weights else {}
        prediction_map = {
            (example.week, example.player_name, example.position): round(
                max(
                    0.0,
                    (
                        applied_weights.get(example.position, 1.0) * prediction
                        + (1 - applied_weights.get(example.position, 1.0))
                        * calculate_weekly_heuristic_projection(example)
                    ),
                ),
                2,
            )
            for example, prediction in zip(weekly_examples, predictions, strict=True)
        }

        return cls(
            training_result,
            prediction_map,
            neural_weights_by_position,
            use_calibrated_weights=use_calibrated_weights,
        )

    def predict_player(
        self,
        player: Player,
        performances: list[WeeklyPlayerPerformance],
        week: int,
    ) -> float:
        prediction = self.predictions.get((week, player.name, player.position))

        if prediction is not None:
            return prediction

        history = get_player_history_before_week(performances, player, week)
        return calculate_weekly_projection(player, history)

    def project_roster(
        self,
        roster: list[Player],
        performances: list[WeeklyPlayerPerformance],
        week: int,
    ) -> list[Player]:
        return [
            replace(
                player,
                projected_score=self.predict_player(player, performances, week),
            )
            for player in roster
        ]


def load_weekly_projection_service(
    model_path: Path = DEFAULT_WEEKLY_MODEL_PATH,
    target_season: int | None = None,
    raw_data_dir=RAW_DATA_DIR,
    use_calibrated_weights: bool = False,
) -> WeeklyNeuralProjectionService | None:
    if target_season is None or not model_path.exists():
        return None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    checkpoint_features = tuple(checkpoint.get("feature_names", ()))

    if checkpoint_features and checkpoint_features != WEEKLY_PROJECTION_FEATURE_NAMES:
        return None

    max_training_season = checkpoint.get("max_training_season")

    if max_training_season is None:
        return None

    if target_season <= int(max_training_season):
        return None

    return WeeklyNeuralProjectionService.from_checkpoint(
        model_path=model_path,
        target_season=target_season,
        raw_data_dir=raw_data_dir,
        use_calibrated_weights=use_calibrated_weights,
    )
