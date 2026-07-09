from evolution.genome import DraftStrategyGenome
from evolution.training import (
    DEFAULT_GENERATION_COUNT,
    DEFAULT_POPULATION_SIZE,
    GenerationResult,
    TrainingResult,
    calculate_generation_mutation_strength,
    format_generation_roster_report,
    format_training_log,
    get_roster_signature,
    have_same_winning_roster,
    run_and_save_training_experiment,
    run_training_experiment,
    save_best_genome,
)
from fantasy_engine.fake_data import create_fake_player_pool
from fantasy_engine.league import League
from fantasy_engine.team import Team


def create_test_league() -> League:
    teams = [Team(name=f"Team {number}") for number in range(1, 11)]

    return League(
        name="Test League",
        teams=teams,
        available_players=create_fake_player_pool(),
    )


def test_default_training_constants_match_task_size():
    assert DEFAULT_POPULATION_SIZE == 100
    assert DEFAULT_GENERATION_COUNT == 20


def test_run_training_experiment_returns_result_for_each_generation():
    league = create_test_league()

    training_result = run_training_experiment(
        league=league,
        population_size=5,
        generation_count=3,
        selection_count=2,
        seed=1,
    )

    assert len(training_result.generation_results) == 3


def test_run_training_experiment_tracks_best_score_and_genome():
    league = create_test_league()

    training_result = run_training_experiment(
        league=league,
        population_size=5,
        generation_count=3,
        selection_count=2,
        seed=1,
    )

    assert training_result.best_score > 0.0
    assert isinstance(training_result.best_genome, DraftStrategyGenome)


def test_generation_results_include_generation_number_and_best_score():
    league = create_test_league()

    training_result = run_training_experiment(
        league=league,
        population_size=5,
        generation_count=3,
        selection_count=2,
        seed=1,
    )

    first_generation_result = training_result.generation_results[0]

    assert first_generation_result.generation_number == 1
    assert first_generation_result.best_score > 0.0
    assert isinstance(first_generation_result.best_genome, DraftStrategyGenome)


def test_format_training_log_includes_every_generation():
    first_genome = DraftStrategyGenome(
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
    second_genome = DraftStrategyGenome(
        projection_weight=0.6,
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

    training_result = TrainingResult(
        generation_results=[
            GenerationResult(
                generation_number=1,
                best_score=100.0,
                best_genome=first_genome,
                winning_team_name="Team 1",
                winning_roster=[],
            ),
            GenerationResult(
                generation_number=2,
                best_score=120.0,
                best_genome=second_genome,
                winning_team_name="Team 1",
                winning_roster=[],
            ),
        ],
        best_score=120.0,
        best_genome=second_genome,
    )

    training_log = format_training_log(training_result)

    assert "Generation 1: best score = 100.0" in training_log
    assert "Generation 2: best score = 120.0" in training_log
    assert "Overall best score: 120.0" in training_log


def test_save_best_genome_writes_json_file(tmp_path):
    output_path = tmp_path / "best_genome.json"
    genome = DraftStrategyGenome(
        projection_weight=0.6,
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
    training_result = TrainingResult(
        generation_results=[],
        best_score=120.0,
        best_genome=genome,
    )

    saved_path = save_best_genome(training_result, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert DraftStrategyGenome.from_json(output_path.read_text()) == genome


def test_run_and_save_training_experiment_saves_best_genome(tmp_path):
    league = create_test_league()
    output_path = tmp_path / "best_genome.json"

    training_result = run_and_save_training_experiment(
        league=league,
        output_path=output_path,
        population_size=5,
        generation_count=3,
        selection_count=2,
        seed=1,
    )

    assert training_result.best_score > 0.0
    assert output_path.exists()


def test_calculate_generation_mutation_strength_decays_over_time():
    first_strength = calculate_generation_mutation_strength(
        generation_number=1,
        generation_count=20,
        initial_mutation_strength=0.12,
        final_mutation_strength=0.03,
    )
    last_strength = calculate_generation_mutation_strength(
        generation_number=20,
        generation_count=20,
        initial_mutation_strength=0.12,
        final_mutation_strength=0.03,
    )

    assert first_strength == 0.12
    assert last_strength == 0.03


def test_training_result_tracks_winning_roster():
    league = create_test_league()

    training_result = run_training_experiment(
        league=league,
        population_size=5,
        generation_count=3,
        selection_count=2,
        seed=1,
    )

    first_generation = training_result.generation_results[0]

    assert first_generation.winning_team_name != ""
    assert len(first_generation.winning_roster) == 16


def test_get_roster_signature_matches_same_players():
    league = create_test_league()

    training_result = run_training_experiment(
        league=league,
        population_size=5,
        generation_count=3,
        selection_count=2,
        seed=1,
    )

    first_generation = training_result.generation_results[0]

    assert get_roster_signature(first_generation.winning_roster)


def test_have_same_winning_roster_returns_boolean():
    league = create_test_league()

    training_result = run_training_experiment(
        league=league,
        population_size=5,
        generation_count=3,
        selection_count=2,
        seed=1,
    )

    same_roster = have_same_winning_roster(training_result, [1, 2, 3])

    assert isinstance(same_roster, bool)


def test_format_generation_roster_report_includes_same_roster_result():
    league = create_test_league()

    training_result = run_training_experiment(
        league=league,
        population_size=5,
        generation_count=3,
        selection_count=2,
        seed=1,
    )

    report = format_generation_roster_report(training_result, [1, 2, 3])

    assert "Generation 1:" in report
    assert "Same winning roster:" in report
