import pytest

from scripts.train_cuda_manager_policy import validate_season_window


def test_cuda_training_requires_holdout_after_training_window():
    with pytest.raises(ValueError, match="after every training season"):
        validate_season_window(2021, 2024, 2024)


def test_cuda_training_rejects_inverted_season_window():
    with pytest.raises(ValueError, match="end-season"):
        validate_season_window(2024, 2021, 2025)


def test_cuda_training_allows_zero_to_disable_holdout():
    validate_season_window(2021, 2024, 0)


def test_cuda_training_rejects_negative_holdout():
    with pytest.raises(ValueError, match="zero or a positive"):
        validate_season_window(2021, 2024, -1)
