from fantasy_engine.player import Player
from models.projection_ensemble import ProjectionEnsemble, combine_predictions


class FixedService:
    def __init__(self, value: float):
        self.value = value

    def predict_player(self, player: Player) -> float:
        return self.value


def test_combine_predictions_reports_uncertainty():
    prediction = combine_predictions([10.0, 14.0])

    assert prediction.expected_points == 12.0
    assert prediction.uncertainty == 2.0
    assert prediction.lower_bound == 10.0
    assert prediction.upper_bound == 14.0


def test_projection_ensemble_predicts_and_projects_player():
    ensemble = ProjectionEnsemble([FixedService(10.0), FixedService(14.0)])
    player = Player(name="Player", position="RB", team="TEST")

    prediction = ensemble.predict_player(player)
    projected = ensemble.project_player(player)

    assert prediction.expected_points == 12.0
    assert projected.projected_score == 12.0
