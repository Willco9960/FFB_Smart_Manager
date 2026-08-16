from models.projection_metrics import (
    calibration_error,
    lineup_regret,
    quantile_coverage,
    spearman_rank_correlation,
    top_k_hit_rate,
)


def test_rank_and_top_k_metrics_reward_correct_order():
    predictions = [3.0, 2.0, 1.0]
    targets = [30.0, 20.0, 10.0]
    assert spearman_rank_correlation(predictions, targets) == 1.0
    assert top_k_hit_rate(predictions, targets, 2) == 1.0


def test_quantile_coverage_and_calibration_are_bounded():
    assert quantile_coverage([0.0, 2.0], [2.0, 4.0], [1.0, 5.0]) == 0.5
    error = calibration_error([0.1, 0.9], [False, True])
    assert 0.0 <= error <= 1.0


def test_lineup_regret_is_zero_for_optimal_selection():
    assert lineup_regret([3.0, 2.0, 1.0], [30.0, 20.0, 10.0], lineup_size=2) == 0.0
