from fantasy_engine.player import Player
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from models.weekly_projection_service import WeeklyNeuralProjectionService


def test_service_uses_neural_prediction_for_matching_weekly_player():
    service = WeeklyNeuralProjectionService(
        training_result=None,
        predictions={(2, "Test RB", "RB"): 18.5},
    )
    player = Player(
        name="Test RB",
        position="RB",
        team="ATL",
        projected_score=240.0,
    )

    prediction = service.predict_player(player, [], week=2)

    assert prediction == 18.5


def test_service_falls_back_to_existing_heuristic_for_unseen_player():
    service = WeeklyNeuralProjectionService(
        training_result=None,
        predictions={},
    )
    player = Player(
        name="Test RB",
        position="RB",
        team="ATL",
        projected_score=210.0,
    )
    performances = [
        WeeklyPlayerPerformance(
            player_id="test-rb",
            player_name="Test RB",
            position="RB",
            team="ATL",
            week=1,
            fantasy_points=20.0,
        )
    ]

    prediction = service.predict_player(player, performances, week=2)

    assert prediction == 18.5
