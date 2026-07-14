from pathlib import Path

import torch

from fantasy_engine.player import Player
from models.draft_projection_nn import DraftProjectionNetwork
from models.projection_service import NeuralProjectionService, load_neural_projection_service


def create_service(tmp_path: Path) -> NeuralProjectionService:
    model = DraftProjectionNetwork(input_size=7)
    checkpoint_path = tmp_path / "projection.pt"
    torch.save(
        {
            "input_size": 7,
            "state_dict": model.state_dict(),
            "feature_means": (100.0, 100.0, 0.0, 0.25, 0.25, 0.25, 0.25),
            "feature_standard_deviations": (50.0, 50.0, 1.0, 0.43, 0.43, 0.43, 0.43),
            "target_mean": 150.0,
            "target_standard_deviation": 75.0,
        },
        checkpoint_path,
    )

    return NeuralProjectionService.from_checkpoint(checkpoint_path)


def test_service_uses_training_feature_order(tmp_path):
    service = create_service(tmp_path)
    player = Player(name="Test QB", position="QB", team="TEST", projected_score=200.0)

    assert service.create_features(player) == (200.0, 200.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def test_service_predicts_nonnegative_player_projection(tmp_path):
    service = create_service(tmp_path)
    player = Player(name="Test RB", position="RB", team="TEST", projected_score=100.0)

    assert service.predict_player(player) >= 0.0


def test_missing_checkpoint_returns_none(tmp_path):
    assert load_neural_projection_service(tmp_path / "missing.pt") is None


def test_target_season_requires_training_metadata(tmp_path):
    checkpoint_path = tmp_path / "projection.pt"
    model = DraftProjectionNetwork(input_size=7)
    torch.save(
        {
            "input_size": 7,
            "state_dict": model.state_dict(),
            "feature_means": (100.0, 100.0, 0.0, 0.25, 0.25, 0.25, 0.25),
            "feature_standard_deviations": (50.0, 50.0, 1.0, 0.43, 0.43, 0.43, 0.43),
            "target_mean": 150.0,
            "target_standard_deviation": 75.0,
        },
        checkpoint_path,
    )

    assert load_neural_projection_service(checkpoint_path, target_season=2021) is None
