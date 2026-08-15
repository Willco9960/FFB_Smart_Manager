import pytest
import torch

from fantasy_engine.player import Player
from gpu_sim.tensor_state import TensorScenarioBatch, create_synthetic_scenario_batch


def _players(actual_offset: float = 0.0):
    return [
        Player(
            "QB", "QB", "TST", projected_score=10.0, actual_score=11.0 + actual_offset
        ),
        Player(
            "RB", "RB", "TST", projected_score=9.0, actual_score=10.0 + actual_offset
        ),
    ]


def test_tensor_scenario_batch_from_players_preserves_data():
    batch = TensorScenarioBatch.from_players(_players())

    assert batch.projected_points.tolist() == [[10.0, 9.0]]
    assert batch.actual_points.tolist() == [[11.0, 10.0]]
    assert batch.player_keys == (("QB", "QB"), ("RB", "RB"))


def test_tensor_scenario_batch_rejects_mismatched_player_order():
    with pytest.raises(ValueError, match="identical player ordering"):
        TensorScenarioBatch.from_player_scenarios(
            [_players(), list(reversed(_players()))]
        )


def test_tensor_scenario_batch_moves_all_tensors_together():
    batch = TensorScenarioBatch.from_players(_players())
    moved = batch.to("cpu")

    assert moved.projected_points.device.type == "cpu"
    assert moved.actual_points.device.type == "cpu"
    assert moved.positions.device.type == "cpu"


def test_synthetic_batch_is_reproducible():
    first = create_synthetic_scenario_batch(3, 12)
    second = create_synthetic_scenario_batch(3, 12)

    assert torch.equal(first.projected_points, second.projected_points)
    assert torch.equal(first.actual_points, second.actual_points)
    assert torch.equal(first.positions, second.positions)
