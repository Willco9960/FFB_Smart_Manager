from evolution.multi_season_evaluation import MultiSeasonEvaluation, SeasonEvaluation


def create_result(season: int, champion: bool = False) -> SeasonEvaluation:
    return SeasonEvaluation(
        season=season,
        fitness=100.0 + season,
        wins=8,
        points_for=900.0,
        playoff_seed=2,
        playoff_wins=1,
        champion=champion,
        transaction_reward=5.0,
        baseline_average_fitness=90.0,
    )


def test_multi_season_evaluation_aggregates_playoffs_and_championships():
    result = MultiSeasonEvaluation((create_result(2021, True), create_result(2022)))

    assert result.average_fitness == (2121.0 + 2122.0) / 2
    assert result.playoff_count == 2
    assert result.championship_count == 1
