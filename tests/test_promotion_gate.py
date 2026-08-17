import pytest

from evolution.promotion_gate import evaluate_promotion_gate


def test_promotion_gate_requires_positive_unseen_season_interval():
    decision = evaluate_promotion_gate(
        candidate_fitness=[120.0, 130.0, 140.0],
        baseline_fitness=[100.0, 100.0, 100.0],
        candidate_wins=[8.0, 9.0, 10.0],
        baseline_wins=[7.0, 7.0, 7.0],
    )
    assert decision.promoted is True
    assert decision.lower_delta > 0.0


def test_promotion_gate_rejects_regression():
    decision = evaluate_promotion_gate(
        candidate_fitness=[90.0, 95.0],
        baseline_fitness=[100.0, 100.0],
        candidate_wins=[6.0, 6.0],
        baseline_wins=[8.0, 8.0],
    )
    assert decision.promoted is False
    assert decision.reasons


def test_promotion_gate_requires_multiple_unseen_seasons():
    with pytest.raises(ValueError, match="at least two"):
        evaluate_promotion_gate(
            candidate_fitness=[120.0],
            baseline_fitness=[100.0],
            candidate_wins=[8.0],
            baseline_wins=[7.0],
        )


def test_promotion_gate_rejects_in_sample_evaluation_season():
    with pytest.raises(ValueError, match="after training"):
        evaluate_promotion_gate(
            candidate_fitness=[120.0, 121.0],
            baseline_fitness=[100.0, 101.0],
            candidate_wins=[8.0, 8.0],
            baseline_wins=[7.0, 7.0],
            evaluation_seasons=(2024, 2025),
            training_end_season=2024,
        )
