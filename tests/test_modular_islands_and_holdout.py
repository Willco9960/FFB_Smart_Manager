from evolution.island_policy_training import (
    IslandSegmentResult,
    train_island_policy_self_play,
)
from evolution.modular_holdout import ModularHoldoutResult
from models.modular_manager_policy import ModularManagerPolicyNetwork


def test_holdout_result_is_json_ready():
    result = ModularHoldoutResult(
        label="candidate",
        season=2025,
        fitness=100.0,
        wins=8.0,
        points_for=900.0,
        playoff_rate=0.75,
        championship_rate=0.25,
        transaction_reward=2.0,
    )

    assert result.to_dict()["season"] == 2025
    assert result.to_dict()["playoff_rate"] == 0.75


def test_island_training_uses_segment_barriers_and_ring_migration(monkeypatch):
    policies_seen = []

    def fake_run(payload):
        island_index = payload[0]
        segment_number = payload[1]
        initial_policy = payload[2]
        policies_seen.append((segment_number, island_index, initial_policy))
        return IslandSegmentResult(
            island_index=island_index,
            segment_number=segment_number,
            best_score=float((segment_number * 10) + island_index),
            policy=initial_policy,
            history=[float((segment_number * 10) + island_index)],
        )

    monkeypatch.setattr(
        "evolution.island_policy_training._run_island_segment",
        fake_run,
    )
    result = train_island_policy_self_play(
        initial_policy=ModularManagerPolicyNetwork(),
        scenarios=[("scenario", [])],
        transaction_genome=None,
        island_count=2,
        segments=2,
        generations_per_segment=1,
        population_size=2,
        selection_count=1,
        island_workers=1,
    )

    assert result.segment_scores == [[10.0, 11.0], [20.0, 21.0]]
    assert result.best_score == 21.0
    assert len(policies_seen) == 4
    assert policies_seen[2][0] == 2
    assert policies_seen[2][1] == 0
    assert policies_seen[2][2] is not policies_seen[0][2]
