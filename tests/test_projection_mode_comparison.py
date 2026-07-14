from evolution.multi_season_evaluation import SeasonEvaluation
from evolution.projection_mode_comparison import (
    ProjectionComparison,
    ProjectionModeComparison,
    format_projection_comparison,
)


def create_evaluation(fitness: float, points_for: float, champion: bool) -> SeasonEvaluation:
    return SeasonEvaluation(
        season=2021,
        fitness=fitness,
        wins=8,
        points_for=points_for,
        playoff_seed=2,
        playoff_wins=1,
        champion=champion,
        transaction_reward=0.0,
        baseline_average_fitness=200.0,
    )


def test_projection_comparison_calculates_deltas_and_summary():
    result = ProjectionComparison(
        seasons=(
            ProjectionModeComparison(
                season=2021,
                heuristic=create_evaluation(100.0, 900.0, False),
                weekly_neural=create_evaluation(120.0, 950.0, True),
            ),
        )
    )

    assert result.seasons[0].fitness_delta == 20.0
    assert result.seasons[0].points_for_delta == 50.0
    assert result.weekly_neural_championship_count == 1

    report = format_projection_comparison(result)

    assert "weekly NN fitness 120.00" in report
    assert "delta +20.00" in report
