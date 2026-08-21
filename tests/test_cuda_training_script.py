import json

import pytest

from scripts.train_cuda_manager_policy import (
    evaluate_cuda_promotion_readiness,
    load_multi_seed_evidence,
    load_parity_evidence,
    resolve_holdout_seasons,
    validate_opponent_archive_size,
    validate_player_count,
    validate_season_window,
)


def test_cuda_training_requires_holdout_after_training_window():
    with pytest.raises(ValueError, match="after every training season"):
        validate_season_window(2021, 2024, 2024)


def test_cuda_training_rejects_inverted_season_window():
    with pytest.raises(ValueError, match="end-season"):
        validate_season_window(2024, 2021, 2025)


def test_cuda_training_allows_zero_to_disable_holdout():
    validate_season_window(2021, 2024, 0)


def test_cuda_training_manifest_can_disable_holdouts_with_zero_sentinel():
    assert resolve_holdout_seasons(0, (0,)) == (0,)


def test_cuda_training_rejects_negative_holdout():
    with pytest.raises(ValueError, match="zero or a positive"):
        validate_season_window(2021, 2024, -1)


def test_cuda_training_accepts_two_unseen_holdouts():
    holdouts = resolve_holdout_seasons(0, (2024, 2025))
    assert holdouts == (2024, 2025)
    validate_season_window(2001, 2023, holdouts)


def test_cuda_training_rejects_duplicate_holdouts():
    with pytest.raises(ValueError, match="unique"):
        resolve_holdout_seasons(0, (2025, 2025))


def test_cuda_training_rejects_mixed_single_and_multiple_holdouts():
    with pytest.raises(ValueError, match="either"):
        resolve_holdout_seasons(2025, (2024, 2025))


def test_parity_evidence_requires_exact_matching_mode(tmp_path):
    report_path = tmp_path / "parity.json"
    report_path.write_text(
        json.dumps(
            {
                "transactions": False,
                "exact_standings_match": True,
                "exact_champion_match": True,
                "exact_weekly_score_match": True,
                "max_weekly_score_abs_delta": 0.0,
            }
        ),
        encoding="utf-8",
    )

    evidence = load_parity_evidence(
        report_path,
        transactions_enabled=False,
        require_promotion_ready=True,
    )

    assert evidence["promotion_ready"] is True
    with pytest.raises(ValueError, match="does not prove exact"):
        load_parity_evidence(
            report_path,
            transactions_enabled=True,
            require_promotion_ready=True,
        )


def test_parity_evidence_missing_report_blocks_only_when_required():
    assert load_parity_evidence(
        None,
        transactions_enabled=True,
        require_promotion_ready=False,
    )["promotion_ready"] is False
    with pytest.raises(ValueError, match="requires --parity-report"):
        load_parity_evidence(
            None,
            transactions_enabled=True,
            require_promotion_ready=True,
        )


def test_cuda_training_rejects_invalid_opponent_archive_size():
    with pytest.raises(ValueError, match="archive size"):
        validate_opponent_archive_size(0)


def test_cuda_training_rejects_invalid_player_count():
    with pytest.raises(ValueError, match="players"):
        validate_player_count(64)


def test_cuda_training_accepts_valid_player_count():
    validate_player_count(160)


def test_multi_seed_evidence_requires_complete_independent_seed_rows(tmp_path):
    report_path = tmp_path / "multi-seed.json"
    report_path.write_text(json.dumps({
        "promotion_ready_multi_seed": True,
        "seeds": [11, 22],
        "by_season": {"2025": {"seed_count": 1}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="seed count"):
        load_multi_seed_evidence(report_path, require_promotion_ready=True)


def test_multi_seed_evidence_rejects_claimed_readiness_with_nonpositive_rows(tmp_path):
    report_path = tmp_path / "multi-seed.json"
    report_path.write_text(json.dumps({
        "promotion_ready_multi_seed": True,
        "seeds": [11, 22],
        "holdout_seasons": [2024, 2025],
        "transactions": False,
        "rows": [
            {
                "season": 2024,
                "seed": 11,
                "delta_vs_initial": 0.0,
                "risk_adjusted_delta_vs_initial": 0.0,
            },
            {
                "season": 2024,
                "seed": 22,
                "delta_vs_initial": 1.0,
                "risk_adjusted_delta_vs_initial": 1.0,
            },
            {
                "season": 2025,
                "seed": 11,
                "delta_vs_initial": 1.0,
                "risk_adjusted_delta_vs_initial": 1.0,
            },
            {
                "season": 2025,
                "seed": 22,
                "delta_vs_initial": 1.0,
                "risk_adjusted_delta_vs_initial": 1.0,
            },
        ],
        "by_season": {"2024": {"seed_count": 2}, "2025": {"seed_count": 2}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="readiness claim"):
        load_multi_seed_evidence(
            report_path,
            require_promotion_ready=True,
            expected_holdout_seasons=(2024, 2025),
            expected_transactions_enabled=False,
        )


def test_multi_seed_evidence_rejects_initial_policy_hash_mismatch(tmp_path):
    report_path = tmp_path / "multi-seed.json"
    report_path.write_text(json.dumps({
        "promotion_ready_multi_seed": False,
        "seeds": [11],
        "holdout_seasons": [2025],
        "transactions": False,
        "initial_policy_sha256": "report-hash",
        "rows": [],
        "by_season": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="initial-policy hash"):
        load_multi_seed_evidence(
            report_path,
            require_promotion_ready=False,
            expected_initial_policy_sha256="active-hash",
        )


def test_promotion_readiness_requires_two_positive_holdouts_and_multi_seed():
    decision = evaluate_cuda_promotion_readiness(
        parity_ready=True,
        multi_seed_ready=True,
        transactions_disabled=False,
        self_play=True,
        self_play_interval=1,
        holdouts=[
            {
                "candidate_delta_vs_initial": 1.0,
                "candidate_risk_adjusted_delta_vs_initial": 0.5,
                "candidate": {"wins": 8.0},
                "initial_policy": {"wins": 7.0},
            },
            {
                "candidate_delta_vs_initial": 2.0,
                "candidate_risk_adjusted_delta_vs_initial": 0.4,
                "candidate": {"wins": 8.0},
                "initial_policy": {"wins": 7.0},
            },
        ],
    )
    assert decision["promotion_ready"] is True


def test_promotion_readiness_blocks_pending_or_regressing_holdouts():
    decision = evaluate_cuda_promotion_readiness(
        parity_ready=True,
        multi_seed_ready=True,
        transactions_disabled=False,
        self_play=True,
        self_play_interval=1,
        holdouts=[
            {"candidate_delta_vs_initial": 1.0, "candidate_risk_adjusted_delta_vs_initial": 0.5},
        ],
    )
    assert decision["promotion_ready"] is False
    assert "at least two unseen holdout seasons are required" in decision["reasons"]
