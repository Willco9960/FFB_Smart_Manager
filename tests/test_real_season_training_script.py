from scripts.train_manager_policy_real_seasons import HOLDOUT_SEASON, TRAINING_SEASONS


def test_real_season_training_uses_future_holdout():
    assert max(TRAINING_SEASONS) < HOLDOUT_SEASON
