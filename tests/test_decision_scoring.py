import pytest

from agents.decision_scoring import (
    blend_policy_and_anchor_scores,
    bounded_policy_score,
    normalize_scores,
)


def test_normalize_scores_preserves_order_and_bounds():
    normalized = normalize_scores([10.0, 20.0, 30.0])

    assert normalized == [0.0, 0.5, 1.0]


def test_normalize_scores_handles_ties():
    assert normalize_scores([4.0, 4.0]) == [0.5, 0.5]


def test_blend_policy_and_anchor_scores_validates_lengths():
    with pytest.raises(ValueError):
        blend_policy_and_anchor_scores([1.0], [1.0, 2.0])


def test_blend_policy_and_anchor_scores_keeps_policy_primary():
    blended = blend_policy_and_anchor_scores(
        policy_scores=[0.0, 100.0],
        anchor_scores=[100.0, 0.0],
        policy_weight=0.8,
    )

    assert blended[1] > blended[0]


def test_bounded_policy_score_is_safe_for_extreme_outputs():
    assert 0.0 < bounded_policy_score(-1_000_000.0) < 1.0
    assert 0.0 < bounded_policy_score(1_000_000.0) < 1.0
