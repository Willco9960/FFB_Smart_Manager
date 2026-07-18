"""Strict chronological folds for projection and policy evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardFold:
    training_seasons: tuple[int, ...]
    validation_season: int
    test_season: int

    def assert_no_future_leakage(self) -> None:
        if not self.training_seasons:
            raise ValueError("A walk-forward fold requires training seasons.")
        if max(self.training_seasons) >= self.validation_season:
            raise ValueError("Training seasons must precede validation season.")
        if self.validation_season >= self.test_season:
            raise ValueError("Validation season must precede test season.")


def build_walk_forward_folds(
    first_season: int,
    last_season: int,
    minimum_training_seasons: int = 5,
) -> list[WalkForwardFold]:
    if last_season - first_season + 1 < minimum_training_seasons + 2:
        raise ValueError("Not enough seasons to build a walk-forward fold.")

    folds = []
    for validation_season in range(first_season + minimum_training_seasons, last_season):
        training_seasons = tuple(range(first_season, validation_season))
        fold = WalkForwardFold(
            training_seasons=training_seasons,
            validation_season=validation_season,
            test_season=validation_season + 1,
        )
        fold.assert_no_future_leakage()
        folds.append(fold)
    return folds
