import json

from evolution.genome import (
    GENOME_WEIGHT_RANGES,
    DraftStrategyGenome,
    create_random_genome,
    random_weight,
)


def assert_weight_in_expected_range(weight_name: str, value: float):
    minimum, maximum = GENOME_WEIGHT_RANGES[weight_name]

    assert minimum <= value <= maximum


def test_random_weight_returns_value_between_zero_and_one():
    import random

    rng = random.Random(42)

    value = random_weight(rng)

    assert 0.0 <= value <= 1.0


def test_create_random_genome_has_all_strategy_weights_in_expected_ranges():
    genome = create_random_genome(seed=42)

    assert_weight_in_expected_range("projection_weight", genome.projection_weight)
    assert_weight_in_expected_range(
        "position_scarcity_weight",
        genome.position_scarcity_weight,
    )
    assert_weight_in_expected_range("adp_value_weight", genome.adp_value_weight)
    assert_weight_in_expected_range("upside_weight", genome.upside_weight)
    assert_weight_in_expected_range("floor_weight", genome.floor_weight)
    assert_weight_in_expected_range("bye_week_penalty", genome.bye_week_penalty)
    assert_weight_in_expected_range("qb_priority", genome.qb_priority)
    assert_weight_in_expected_range("rb_priority", genome.rb_priority)
    assert_weight_in_expected_range("wr_priority", genome.wr_priority)
    assert_weight_in_expected_range("te_priority", genome.te_priority)


def test_random_genome_starts_with_more_realistic_position_priorities():
    genome = create_random_genome(seed=7)

    assert genome.rb_priority > genome.qb_priority
    assert genome.wr_priority > genome.qb_priority


def test_create_random_genome_is_repeatable_with_seed():
    first_genome = create_random_genome(seed=123)
    second_genome = create_random_genome(seed=123)

    assert first_genome == second_genome


def test_genome_can_convert_to_dict():
    genome = DraftStrategyGenome(
        projection_weight=0.1,
        position_scarcity_weight=0.2,
        adp_value_weight=0.3,
        upside_weight=0.4,
        floor_weight=0.5,
        bye_week_penalty=0.6,
        qb_priority=0.7,
        rb_priority=0.8,
        wr_priority=0.9,
        te_priority=1.0,
    )

    genome_data = genome.to_dict()

    assert genome_data["projection_weight"] == 0.1
    assert genome_data["position_scarcity_weight"] == 0.2
    assert genome_data["adp_value_weight"] == 0.3
    assert genome_data["upside_weight"] == 0.4
    assert genome_data["floor_weight"] == 0.5
    assert genome_data["bye_week_penalty"] == 0.6
    assert genome_data["qb_priority"] == 0.7
    assert genome_data["rb_priority"] == 0.8
    assert genome_data["wr_priority"] == 0.9
    assert genome_data["te_priority"] == 1.0


def test_genome_can_serialize_to_json():
    genome = DraftStrategyGenome(
        projection_weight=0.1,
        position_scarcity_weight=0.2,
        adp_value_weight=0.3,
        upside_weight=0.4,
        floor_weight=0.5,
        bye_week_penalty=0.6,
        qb_priority=0.7,
        rb_priority=0.8,
        wr_priority=0.9,
        te_priority=1.0,
    )

    genome_json = genome.to_json()
    genome_data = json.loads(genome_json)

    assert genome_data["projection_weight"] == 0.1
    assert genome_data["te_priority"] == 1.0


def test_genome_can_load_from_json():
    genome = DraftStrategyGenome(
        projection_weight=0.1,
        position_scarcity_weight=0.2,
        adp_value_weight=0.3,
        upside_weight=0.4,
        floor_weight=0.5,
        bye_week_penalty=0.6,
        qb_priority=0.7,
        rb_priority=0.8,
        wr_priority=0.9,
        te_priority=1.0,
    )

    loaded_genome = DraftStrategyGenome.from_json(genome.to_json())

    assert loaded_genome == genome
