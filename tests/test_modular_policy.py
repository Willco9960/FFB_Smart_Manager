from pathlib import Path

import pytest

from agents.neural_draft_agent import NeuralDraftAgent
from agents.neural_lineup_agent import NeuralLineupAgent
from evolution.modular_behavior_cloning import (
    ModularImitationExample,
    train_modular_behavior_policy,
)
from evolution.modular_policy_training import (
    ModularGenerationMetrics,
    adapt_mutation_for_diversity,
    calculate_policy_population_diversity,
    crossover_modular_policies,
    mutate_modular_policy,
    select_scenarios_for_generation,
)
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from models.league_state_encoder import (
    LEAGUE_STATE_FEATURE_NAMES,
    create_league_state_features,
)
from models.modular_manager_policy import (
    ModularManagerPolicyNetwork,
    create_modular_policy_features,
    load_modular_policy_network,
    save_modular_policy_network,
)


def create_player(name: str, position: str, projection: float) -> Player:
    return Player(name=name, position=position, team="TEST", projected_score=projection)


def test_league_state_feature_count_is_stable():
    team = Team(name="Team", roster=[create_player("QB", "QB", 20.0)])
    features = create_league_state_features(team, [create_player("RB", "RB", 18.0)])

    assert len(features) == len(LEAGUE_STATE_FEATURE_NAMES)


def test_modular_policy_has_specialized_heads():
    team = Team(name="Team")
    player = create_player("RB", "RB", 18.0)
    features = create_modular_policy_features(player, team, [player])
    model = ModularManagerPolicyNetwork()

    assert model.score_draft_action(features) != model.score_lineup_action(features)
    assert isinstance(model.estimate_value(features), float)


def test_modular_policy_can_be_saved_and_loaded(tmp_path: Path):
    path = tmp_path / "modular_policy.pt"
    model = ModularManagerPolicyNetwork()

    save_modular_policy_network(model, path)
    loaded = load_modular_policy_network(path)

    assert isinstance(loaded, ModularManagerPolicyNetwork)
    assert loaded.player_feature_count == model.player_feature_count


def test_modular_policy_mutation_is_reproducible():
    model = ModularManagerPolicyNetwork()
    first = mutate_modular_policy(model, __import__("random").Random(7))
    second = mutate_modular_policy(model, __import__("random").Random(7))

    assert all(
        (first_parameter == second_parameter).all()
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        )
    )


def test_policy_population_diversity_is_zero_for_one_policy():
    assert calculate_policy_population_diversity([ModularManagerPolicyNetwork()]) == 0.0


def test_policy_population_diversity_detects_different_policies():
    first = ModularManagerPolicyNetwork()
    second = mutate_modular_policy(first, __import__("random").Random(7), 0.5)

    assert calculate_policy_population_diversity([first, second]) > 0.0


def test_adapt_mutation_for_diversity_boosts_collapsed_population():
    assert adapt_mutation_for_diversity(0.01, 0.001, 0.002, 1.5) == 0.015


def test_adapt_mutation_for_diversity_keeps_healthy_population_stable():
    assert adapt_mutation_for_diversity(0.01, 0.01, 0.002, 1.5) == 0.01


def test_modular_policy_crossover_preserves_shape():
    first = ModularManagerPolicyNetwork()
    second = ModularManagerPolicyNetwork()
    child = crossover_modular_policies(first, second, __import__("random").Random(1))

    assert isinstance(child, ModularManagerPolicyNetwork)
    assert [tuple(parameter.shape) for parameter in child.parameters()] == [
        tuple(parameter.shape) for parameter in first.parameters()
    ]


def test_behavior_cloning_updates_policy():
    model = ModularManagerPolicyNetwork()
    team = Team(name="Team")
    player = create_player("RB", "RB", 18.0)
    features = create_modular_policy_features(player, team, [player])
    before = model.score_draft_action(features)

    loss = train_modular_behavior_policy(
        model,
        [ModularImitationExample(features=features, target_score=5.0)],
        epochs=3,
    )

    assert loss >= 0.0
    assert model.score_draft_action(features) != before


def test_existing_neural_agents_route_to_modular_heads():
    model = ModularManagerPolicyNetwork()
    players = [
        create_player("QB", "QB", 20.0),
        create_player("RB1", "RB", 18.0),
        create_player("RB2", "RB", 17.0),
        create_player("RB3", "RB", 16.5),
        create_player("WR1", "WR", 16.0),
        create_player("WR2", "WR", 15.0),
        create_player("TE", "TE", 12.0),
    ]
    team = Team(name="Team")

    selected = NeuralDraftAgent(model).choose_player(players, team, None)
    lineup = NeuralLineupAgent(model, lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES).choose_lineup(
        players
    )

    assert selected in players
    assert lineup.players


def test_modular_policy_accepts_legacy_state_feature_width():
    model = ModularManagerPolicyNetwork(state_feature_count=15)
    team = Team(name="Team")
    player = create_player("RB", "RB", 18.0)
    features = create_modular_policy_features(player, team, [player])

    assert isinstance(model.score_draft_action(features), float)


def test_modular_policy_batch_scores_match_scalar_scores():
    model = ModularManagerPolicyNetwork()
    team = Team(name="Team")
    players = [
        create_player("RB1", "RB", 18.0),
        create_player("WR1", "WR", 17.0),
    ]
    features = [create_modular_policy_features(player, team, players) for player in players]

    batch_scores = model.score_decisions(features, "draft")
    scalar_scores = [model.score_draft_action(item) for item in features]

    assert batch_scores == pytest.approx(scalar_scores)


def test_modular_generation_metrics_are_json_ready():
    metrics = ModularGenerationMetrics(
        generation_number=1,
        generation_count=3,
        scenario_count=2,
        neural_population=10,
        baseline_population=10,
        best_fitness=100.0,
        average_fitness=80.0,
        median_fitness=82.0,
        fitness_stddev=5.0,
        best_wins=8.0,
        best_points_for=1000.0,
        best_playoff_rate=1.0,
        best_championship_rate=0.5,
        best_transaction_reward=2.0,
        baseline_average_fitness=90.0,
        baseline_best_fitness=120.0,
        mutation_strength=0.01,
        elapsed_seconds=12.5,
        cumulative_best_fitness=100.0,
        cumulative_best_generation=1,
        scenario_labels=("Scenario 1", "Scenario 2"),
    )

    payload = metrics.to_dict()

    assert payload["generation_number"] == 1
    assert payload["best_playoff_rate"] == 1.0
    assert payload["baseline_best_fitness"] == 120.0


def test_scenario_rotation_is_deterministic_and_wraps():
    scenarios = [(index, []) for index in range(5)]

    selected = select_scenarios_for_generation(
        scenarios=scenarios,
        generation_number=2,
        scenarios_per_generation=2,
    )

    assert selected == [(0, []), (4, [])]


def test_scenario_rotation_runs_full_evaluation_on_interval():
    scenarios = [(index, []) for index in range(5)]

    selected = select_scenarios_for_generation(
        scenarios=scenarios,
        generation_number=4,
        scenarios_per_generation=2,
        full_evaluation_interval=4,
    )

    assert selected == scenarios
