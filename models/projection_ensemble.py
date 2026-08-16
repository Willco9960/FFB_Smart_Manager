"""Prediction ensembles and uncertainty estimates for manager decisions."""

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Protocol

from fantasy_engine.player import Player


class PlayerProjectionService(Protocol):
    def predict_player(self, player: Player) -> float: ...


@dataclass(frozen=True)
class ProjectionPrediction:
    expected_points: float
    uncertainty: float
    lower_bound: float
    upper_bound: float


def combine_predictions(
    predictions: list[float], confidence_z: float = 1.0
) -> ProjectionPrediction:
    if not predictions:
        raise ValueError("At least one prediction is required.")

    expected = mean(predictions)
    uncertainty = pstdev(predictions) if len(predictions) > 1 else 0.0
    return ProjectionPrediction(
        expected_points=round(expected, 2),
        uncertainty=round(uncertainty, 2),
        lower_bound=round(expected - confidence_z * uncertainty, 2),
        upper_bound=round(expected + confidence_z * uncertainty, 2),
    )


class ProjectionEnsemble:
    def __init__(self, services: list[PlayerProjectionService]):
        if not services:
            raise ValueError("At least one projection service is required.")
        self.services = services

    def predict_player(self, player: Player) -> ProjectionPrediction:
        predictions = [service.predict_player(player) for service in self.services]
        return combine_predictions(predictions)

    def project_player(self, player: Player) -> Player:
        prediction = self.predict_player(player)
        return Player(
            name=player.name,
            position=player.position,
            team=player.team,
            projected_score=prediction.expected_points,
            actual_score=player.actual_score,
            player_id=player.player_id,
            history_missing=player.history_missing,
        )
