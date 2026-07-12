from dataclasses import replace
from pathlib import Path

import torch

from fantasy_engine.league import League
from fantasy_engine.player import Player
from models.draft_projection_nn import DraftProjectionNetwork, FeatureScaler


class NeuralProjectionService:
    def __init__(
        self,
        model: DraftProjectionNetwork,
        feature_scaler: FeatureScaler,
        target_mean: float,
        target_standard_deviation: float,
    ):
        self.model = model.cpu()
        self.feature_scaler = feature_scaler
        self.target_mean = target_mean
        self.target_standard_deviation = target_standard_deviation
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, model_path: Path) -> "NeuralProjectionService":
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        model = DraftProjectionNetwork(input_size=checkpoint["input_size"])
        model.load_state_dict(checkpoint["state_dict"])
        feature_scaler = FeatureScaler(
            means=tuple(checkpoint["feature_means"]),
            standard_deviations=tuple(checkpoint["feature_standard_deviations"]),
        )

        return cls(
            model=model,
            feature_scaler=feature_scaler,
            target_mean=float(checkpoint["target_mean"]),
            target_standard_deviation=float(checkpoint["target_standard_deviation"]),
        )

    def create_features(self, player: Player) -> tuple[float, ...]:
        previous_points = float(player.projected_score)
        position_features = (
            float(player.position == "QB"),
            float(player.position == "RB"),
            float(player.position == "WR"),
            float(player.position == "TE"),
        )

        return (
            previous_points,
            previous_points,
            0.0,
            *position_features,
        )

    def predict_player(self, player: Player) -> float:
        if player.position not in {"QB", "RB", "WR", "TE"}:
            return round(max(0.0, player.projected_score), 2)

        features = self.create_features(player)
        scaled_features = [
            (value - mean) / standard_deviation
            for value, mean, standard_deviation in zip(
                features,
                self.feature_scaler.means,
                self.feature_scaler.standard_deviations,
                strict=True,
            )
        ]
        feature_tensor = torch.tensor([scaled_features], dtype=torch.float32)

        with torch.no_grad():
            normalized_prediction = self.model(feature_tensor).item()

        prediction = (
            normalized_prediction * self.target_standard_deviation
        ) + self.target_mean

        return round(max(0.0, prediction), 2)

    def project_player(self, player: Player) -> Player:
        return replace(player, projected_score=self.predict_player(player))

    def project_league(self, league: League) -> League:
        return replace(
            league,
            teams=[
                replace(team, roster=[self.project_player(player) for player in team.roster])
                for team in league.teams
            ],
            available_players=[
                self.project_player(player) for player in league.available_players
            ],
        )


def load_neural_projection_service(
    model_path: Path,
) -> NeuralProjectionService | None:
    if not model_path.exists():
        return None

    return NeuralProjectionService.from_checkpoint(model_path)
