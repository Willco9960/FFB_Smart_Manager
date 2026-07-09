import json

from evolution.genome import DraftStrategyGenome, create_random_genome, random_weight


def test_random_weight_returns_value_between_zero_and_one():
    import random

    rng = random.Random(42)

    value = random_weight(rng)

    assert 0.0 <= value <= 1.0


def test_create_random_genome_has_all_strategy_weights():
    genome = create_random_genome(seed=42)

    assert 0.0 <= genome.projection_weight <= 1.0
    assert 0.0 <= genome.position_scarcity_weight <= 1.0
    assert 0.0 <= genome.adp_value_weight <= 1.0
    assert 0.0 <= genome.upside_weight <= 1.0
    assert 0.0 <= genome.floor_weight <= 1.0
    assert 0.0 <= genome.bye_week_penalty <= 1.0
    assert 0.0 <= genome.qb_priority <= 1.0
    assert 0.0 <= genome.rb_priority <= 1.0
    assert 0.0 <= genome.wr_priority <= 1.0
    assert 0.0 <= genome.te_priority <= 1.0


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