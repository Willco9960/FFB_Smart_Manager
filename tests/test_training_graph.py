from evolution.genome import DraftStrategyGenome
from evolution.training import GenerationResult, TrainingResult
from evolution.training_graph import (
    get_average_scores,
    get_best_scores,
    get_generation_numbers,
    save_training_progress_graph,
)


def create_test_genome() -> DraftStrategyGenome:
    return DraftStrategyGenome(
        projection_weight=0.5,
        position_scarcity_weight=0.1,
        adp_value_weight=0.2,
        upside_weight=0.3,
        floor_weight=0.4,
        bye_week_penalty=0.0,
        qb_priority=0.1,
        rb_priority=0.5,
        wr_priority=0.5,
        te_priority=0.2,
    )


def create_test_training_result() -> TrainingResult:
    genome = create_test_genome()

    return TrainingResult(
        generation_results=[
            GenerationResult(
                generation_number=1,
                best_score=100.0,
                best_genome=genome,
                winning_team_name="Team 1",
                winning_roster=[],
                average_score=80.0,
            ),
            GenerationResult(
                generation_number=2,
                best_score=120.0,
                best_genome=genome,
                winning_team_name="Team 2",
                winning_roster=[],
                average_score=90.0,
            ),
        ],
        best_score=120.0,
        best_genome=genome,
    )


def test_get_generation_numbers_returns_generation_numbers():
    training_result = create_test_training_result()

    generation_numbers = get_generation_numbers(training_result)

    assert generation_numbers == [1, 2]


def test_get_best_scores_returns_best_scores():
    training_result = create_test_training_result()

    best_scores = get_best_scores(training_result)

    assert best_scores == [100.0, 120.0]


def test_get_average_scores_returns_average_scores():
    training_result = create_test_training_result()

    average_scores = get_average_scores(training_result)

    assert average_scores == [80.0, 90.0]


def test_get_average_scores_returns_none_when_average_missing():
    genome = create_test_genome()
    training_result = TrainingResult(
        generation_results=[
            GenerationResult(
                generation_number=1,
                best_score=100.0,
                best_genome=genome,
                winning_team_name="Team 1",
                winning_roster=[],
            ),
        ],
        best_score=100.0,
        best_genome=genome,
    )

    average_scores = get_average_scores(training_result)

    assert average_scores is None


def test_save_training_progress_graph_creates_file(tmp_path):
    training_result = create_test_training_result()
    output_path = tmp_path / "training_progress.png"

    saved_path = save_training_progress_graph(
        training_result=training_result,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()
