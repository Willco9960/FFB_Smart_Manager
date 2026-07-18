import pytest

from evolution.walk_forward import WalkForwardFold, build_walk_forward_folds


def test_build_walk_forward_folds_are_chronological():
    folds = build_walk_forward_folds(2001, 2010, minimum_training_seasons=5)

    assert folds[0].training_seasons == (2001, 2002, 2003, 2004, 2005)
    assert folds[0].validation_season == 2006
    assert folds[0].test_season == 2007
    assert all(
        max(fold.training_seasons) < fold.validation_season < fold.test_season for fold in folds
    )


def test_walk_forward_fold_rejects_future_training_data():
    fold = WalkForwardFold(
        training_seasons=(2001, 2002, 2007),
        validation_season=2006,
        test_season=2008,
    )

    with pytest.raises(ValueError):
        fold.assert_no_future_leakage()


def test_walk_forward_requires_enough_seasons():
    with pytest.raises(ValueError):
        build_walk_forward_folds(2020, 2024, minimum_training_seasons=5)
