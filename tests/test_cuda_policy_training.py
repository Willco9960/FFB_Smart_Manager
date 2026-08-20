import random
from dataclasses import replace

import pytest
import torch

from evolution.opponent_archive import OpponentArchive
from fantasy_engine.fitness_contract import ESPN_FITNESS_CONTRACT
from gpu_sim.full_season import create_synthetic_season_state, run_full_cuda_season
from gpu_sim.historical_adapter import create_historical_cuda_inputs
from gpu_sim.policy_training import (
    _candidate_auxiliary_metrics,
    _candidate_fitness,
    evaluate_cuda_policy,
    evaluate_cuda_policy_population,
    mutate_policy,
    prepare_cuda_scenario_bank,
    save_cuda_training_state,
    select_training_season_indices,
    summarize_cuda_throughput,
    train_cuda_policy_population,
    validate_cuda_training_state_contract,
    validate_cuda_training_state_manifest,
)
from gpu_sim.tensorized_draft import run_batched_roster_aware_draft
from models.modular_manager_policy import ModularManagerPolicyNetwork


def test_policy_conditioned_draft_runs_against_baselines():
    state = create_synthetic_season_state(
        scenarios=2,
        players=160,
        weeks=17,
        device=torch.device("cpu"),
    )
    policy = ModularManagerPolicyNetwork()

    run_full_cuda_season(
        state,
        policy_network=policy,
        policy_team_indices=torch.tensor([0, 1]),
        enable_transactions=False,
    )

    assert state.rosters is not None
    assert state.rosters.shape == (2, 10, 16)
    assert state.champions is not None


def test_candidate_fitness_attribution_selects_routed_team_only():
    state = create_synthetic_season_state(
        scenarios=1,
        players=160,
        team_count=2,
        weeks=17,
        device=torch.device("cpu"),
    )
    state.wins.zero_()
    state.points_for.zero_()
    state.playoff_wins = torch.zeros_like(state.wins)
    state.champions = torch.full((1,), -1, dtype=torch.long)
    state.waiver_policy_gains = [torch.tensor([[10.0, 1.0]])]
    state.trade_policy_gains = [torch.tensor([[5.0, 2.0]])]
    state.lineup_policy_gains = [torch.tensor([[3.0, 4.0]])]

    fitness, *_ = _candidate_fitness(state, torch.tensor([0]))
    transaction_reward, lineup_efficiency = _candidate_auxiliary_metrics(
        state, torch.tensor([0])
    )

    assert transaction_reward.tolist() == [15.0]
    assert lineup_efficiency.tolist() == [3.0]
    assert fitness.tolist() == pytest.approx([
        15.0 * ESPN_FITNESS_CONTRACT.transaction_reward_weight
        + 3.0 * ESPN_FITNESS_CONTRACT.lineup_efficiency_weight
        + ESPN_FITNESS_CONTRACT.playoff_qualification_reward
    ])


def test_cuda_training_fails_closed_for_incomplete_transaction_contract():
    state = create_synthetic_season_state(
        scenarios=1,
        players=160,
        weeks=17,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="fitness contract is incomplete"):
        train_cuda_policy_population(
            ModularManagerPolicyNetwork(),
            [state],
            population_size=2,
            generations=1,
            selection_count=1,
            scenario_repeats=1,
            enable_transactions=True,
            require_complete_fitness_contract=True,
        )


def test_projection_baseline_draft_satisfies_full_espn_roster_minimums():
    state = create_historical_cuda_inputs(
        season=2024,
        players=256,
        device=torch.device("cpu"),
    ).state

    result = run_batched_roster_aware_draft(
        state.draft_projections,
        state.positions,
        team_count=10,
        rounds=16,
        position_minimums=(1, 4, 4, 1, 1, 1),
        position_maximums=(2, 6, 7, 3, 1, 1),
    )
    counts = torch.stack(
        [(state.positions[result.player_indices] == position).sum(dim=2) for position in range(6)],
        dim=2,
    )

    assert torch.all(counts >= torch.tensor([1, 4, 4, 1, 1, 1]))


def test_cuda_policy_evaluation_returns_projection_baseline_metrics():
    state = create_historical_cuda_inputs(
        season=2024,
        players=256,
        device=torch.device("cpu"),
    ).state

    evaluation = evaluate_cuda_policy(
        None,
        [state],
        scenario_repeats=2,
        projection_noise=0.0,
        enable_transactions=False,
    )

    assert evaluation.fitness_stddev == 0.0
    assert evaluation.wins >= 0.0
    assert evaluation.points_for >= 0.0


def test_cuda_policy_evaluation_returns_fitness_metrics():
    state = create_synthetic_season_state(
        scenarios=2,
        players=160,
        weeks=17,
        device=torch.device("cpu"),
    )
    evaluation = evaluate_cuda_policy(
        ModularManagerPolicyNetwork(),
        [state],
        scenario_repeats=1,
        enable_transactions=False,
    )

    assert evaluation.fitness >= 0.0
    assert evaluation.wins >= 0.0
    assert 0.0 <= evaluation.playoff_rate <= 1.0


def test_self_play_interval_must_be_positive():
    state = create_synthetic_season_state(
        scenarios=1,
        players=160,
        weeks=17,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="self_play_interval"):
        train_cuda_policy_population(
            ModularManagerPolicyNetwork(),
            [state],
            population_size=2,
            generations=1,
            selection_count=1,
            scenario_repeats=1,
            enable_transactions=False,
            self_play=True,
            self_play_interval=0,
        )


def test_cuda_policy_population_training_emits_generation_metrics():
    state = create_synthetic_season_state(
        scenarios=1,
        players=160,
        weeks=17,
        device=torch.device("cpu"),
    )
    _, metrics = train_cuda_policy_population(
        ModularManagerPolicyNetwork(),
        [state],
        population_size=2,
        generations=1,
        selection_count=1,
        scenario_repeats=1,
        enable_transactions=False,
    )

    assert len(metrics) == 1
    assert metrics[0].generation == 1
    assert metrics[0].best_fitness >= 0.0


def test_opponent_archive_round_trip_preserves_policy_and_rating():
    policy = ModularManagerPolicyNetwork()
    archive = OpponentArchive(max_size=4)
    entry = archive.add(policy, label="elite")
    entry.rating = 1532.5

    restored = OpponentArchive.from_state_dict(
        archive.to_state_dict(),
        ModularManagerPolicyNetwork(),
        torch.device("cpu"),
    )

    assert len(restored.entries) == 1
    assert restored.entries[0].label == "elite"
    assert restored.entries[0].rating == 1532.5
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            policy.state_dict().values(),
            restored.entries[0].policy.state_dict().values(),
            strict=True,
        )
    )


def test_cuda_policy_self_play_uses_explicit_opponent_policies():
    state = create_synthetic_season_state(
        scenarios=1,
        players=160,
        weeks=17,
        device=torch.device("cpu"),
    )
    policies = [ModularManagerPolicyNetwork() for _ in range(2)]

    _, metrics = train_cuda_policy_population(
        policies[0],
        [state],
        population_size=2,
        generations=1,
        selection_count=1,
        scenario_repeats=1,
        enable_transactions=False,
        self_play=True,
    )

    assert len(metrics) == 1


def test_adapter_mutation_preserves_shared_encoders():
    policy = ModularManagerPolicyNetwork()
    mutated = mutate_policy(policy, random.Random(3), strength=0.1, adapter_only=True)
    assert torch.equal(
        policy.player_encoder[0].weight,
        mutated.player_encoder[0].weight,
    )
    assert torch.equal(
        policy.value_head[0].weight,
        mutated.value_head[0].weight,
    )


def test_resume_manifest_rejects_changed_search_configuration():
    manifest = {"training_seasons": [2021, 2022], "scenario_repeats": 4}
    validate_cuda_training_state_manifest({"run_manifest": manifest}, manifest)
    with pytest.raises(ValueError, match="scenario_repeats"):
        validate_cuda_training_state_manifest(
            {"run_manifest": manifest},
            {"training_seasons": [2021, 2022], "scenario_repeats": 8},
        )


def test_resume_manifest_rejects_legacy_checkpoint_without_provenance():
    with pytest.raises(ValueError, match="missing run_manifest"):
        validate_cuda_training_state_manifest({}, {"schema_version": 1})


def test_episodic_replay_keeps_oldest_and_newest_seasons():
    assert select_training_season_indices(5, 2, generation=1, replay_interval=3) == [0, 4]
    assert select_training_season_indices(5, 2, generation=2, replay_interval=3) == [0, 4]
    assert select_training_season_indices(5, 2, generation=3, replay_interval=3) == [0, 1, 2, 3, 4]


def test_throughput_summary_reports_stable_gph_and_normalized_scenarios():
    metrics = [
        type("Metric", (), {"generation": 1, "elapsed_seconds": 10.0})(),
        type("Metric", (), {"generation": 2, "elapsed_seconds": 22.0})(),
        type("Metric", (), {"generation": 3, "elapsed_seconds": 34.0})(),
    ]
    report = summarize_cuda_throughput(
        metrics, population=4, training_seasons=5, scenario_repeats=2
    )
    assert report["stable_generations_per_hour"] > 0
    assert report["normalized_scenario_evaluations_per_hour"] > 0
    assert (
        report["stable_generations_per_hour_range"][0]
        <= report["stable_generations_per_hour_range"][1]
    )


def test_cuda_evaluation_uncertainty_includes_repeated_scenarios(monkeypatch):
    state = create_synthetic_season_state(
        scenarios=1,
        players=160,
        weeks=17,
        device=torch.device("cpu"),
    )
    import gpu_sim.policy_training as policy_training

    observed = {}
    real_prepare = policy_training.prepare_cuda_scenario_bank

    def prepare_with_observation(states, **kwargs):
        observed.update(kwargs)
        return real_prepare(states, **kwargs)

    monkeypatch.setattr(policy_training, "prepare_cuda_scenario_bank", prepare_with_observation)
    evaluation = evaluate_cuda_policy(
        ModularManagerPolicyNetwork(),
        [state],
        scenario_repeats=3,
        projection_noise=0.25,
        seed=41,
        enable_transactions=False,
    )
    assert observed["scenario_repeats"] == 3
    assert evaluation.fitness_stddev >= 0.0
    assert evaluation.risk_adjusted_fitness <= evaluation.fitness


def test_full_policy_mutation_can_evolve_shared_encoders():
    policy = ModularManagerPolicyNetwork()
    mutated = mutate_policy(policy, random.Random(3), strength=0.1, adapter_only=False)
    assert not torch.equal(
        policy.player_encoder[0].weight,
        mutated.player_encoder[0].weight,
    )


def test_resume_checkpoint_requires_matching_fitness_contract():
    with pytest.raises(ValueError, match="missing fitness_contract_digest"):
        validate_cuda_training_state_contract({})


def test_cuda_policy_training_checkpoint_resumes_population(tmp_path):
    state = create_synthetic_season_state(
        scenarios=1,
        players=160,
        weeks=17,
        device=torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "training_state.pt"
    policy = ModularManagerPolicyNetwork()

    def checkpoint_callback(generation, population, best_policy, metrics, rng, opponent_archive):
        save_cuda_training_state(
            checkpoint_path,
            generation=generation,
            population=population,
            best_policy=best_policy,
            metrics=metrics,
            rng_state=rng.getstate(),
            opponent_archive=opponent_archive,
        )

    _, first_metrics = train_cuda_policy_population(
        policy,
        [state],
        population_size=2,
        generations=1,
        selection_count=1,
        scenario_repeats=1,
        enable_transactions=False,
        seed=19,
        checkpoint_callback=checkpoint_callback,
        self_play=True,
    )
    resume_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert resume_state["opponent_archive"]["entries"]
    _, resumed_metrics = train_cuda_policy_population(
        policy,
        [state],
        population_size=2,
        generations=2,
        selection_count=1,
        scenario_repeats=1,
        enable_transactions=False,
        seed=19,
        resume_state=resume_state,
        self_play=True,
    )

    assert len(first_metrics) == 1
    assert len(resumed_metrics) == 2
    assert resumed_metrics[0].generation == 1
    assert resumed_metrics[1].generation == 2
    assert resumed_metrics[1].elapsed_seconds > resumed_metrics[0].elapsed_seconds


def test_cuda_training_checkpoint_preserves_previous_file_on_save_failure(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "training_state.pt"
    checkpoint_path.write_bytes(b"previous checkpoint")
    policy = ModularManagerPolicyNetwork()
    real_torch_save = torch.save

    def save_then_fail(payload, target):
        real_torch_save(payload, target)
        raise RuntimeError("simulated checkpoint interruption")

    monkeypatch.setattr(torch, "save", save_then_fail)
    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        save_cuda_training_state(
            checkpoint_path,
            generation=1,
            population=[policy],
            best_policy=policy,
            metrics=[],
            rng_state=random.Random(1).getstate(),
        )

    assert checkpoint_path.read_bytes() == b"previous checkpoint"
    assert not checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp").exists()


def test_batched_population_evaluation_matches_sequential_evaluation():
    state = create_synthetic_season_state(
        scenarios=2,
        players=160,
        weeks=17,
        device=torch.device("cpu"),
    )
    policies = [ModularManagerPolicyNetwork() for _ in range(2)]
    scenario_bank = prepare_cuda_scenario_bank(
        [state],
        scenario_repeats=1,
        projection_noise=0.0,
        seed=7,
    )
    sequential = [
        evaluate_cuda_policy(
            policy,
            [state],
            scenario_bank=scenario_bank,
            enable_transactions=False,
        )
        for policy in policies
    ]
    batched = evaluate_cuda_policy_population(
        policies,
        [state],
        scenario_bank=scenario_bank,
        enable_transactions=False,
    )

    for expected, actual in zip(sequential, batched, strict=True):
        assert actual.fitness == expected.fitness
        assert actual.wins == expected.wins
        assert actual.points_for == expected.points_for


def test_batched_population_transaction_heads_match_exact_evaluation():
    state = create_synthetic_season_state(
        scenarios=1,
        players=160,
        weeks=17,
        device=torch.device("cpu"),
    )
    policies = [ModularManagerPolicyNetwork() for _ in range(2)]
    scenario_bank = prepare_cuda_scenario_bank(
        [state],
        scenario_repeats=1,
        projection_noise=0.0,
        seed=11,
    )
    exact = evaluate_cuda_policy_population(
        policies,
        [state],
        scenario_bank=scenario_bank,
        enable_transactions=True,
        exact_policy_head_parity=True,
    )
    batched = evaluate_cuda_policy_population(
        policies,
        [state],
        scenario_bank=scenario_bank,
        enable_transactions=True,
        exact_policy_head_parity=False,
    )

    for expected, actual in zip(exact, batched, strict=True):
        assert actual.fitness == expected.fitness
        assert actual.wins == expected.wins
        assert actual.points_for == expected.points_for
        assert actual.playoff_rate == expected.playoff_rate
        assert actual.championship_rate == expected.championship_rate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_batched_population_moves_cpu_policies_to_cuda_state_device():
    state = create_synthetic_season_state(
        scenarios=1,
        players=160,
        weeks=17,
        device=torch.device("cuda"),
    )
    policies = [ModularManagerPolicyNetwork() for _ in range(2)]
    scenario_bank = prepare_cuda_scenario_bank(
        [state],
        scenario_repeats=1,
        projection_noise=0.0,
        seed=23,
    )

    evaluations = evaluate_cuda_policy_population(
        policies,
        [state],
        scenario_bank=scenario_bank,
        enable_transactions=False,
        exact_policy_head_parity=False,
    )

    assert len(evaluations) == len(policies)


def test_batched_population_preserves_contract_and_distributional_context():
    state = create_synthetic_season_state(
        scenarios=1,
        players=160,
        weeks=17,
        device=torch.device("cpu"),
    )
    state.draft_floors = state.draft_projections * 0.8
    state.draft_medians = state.draft_projections
    state.draft_ceilings = state.draft_projections * 1.2
    state.draft_boom_probabilities = torch.full_like(state.draft_projections, 0.25)
    state.positions = torch.tensor(
        ([0, 1, 1, 2, 2, 3, 4, 5] * 20)[:160],
        dtype=torch.long,
    )
    state.lineup_position_rules = (
        (0,),
        (1,),
        (1,),
        (2,),
        (2,),
        (3,),
        (1, 2, 3),
        (4,),
        (5,),
    )
    contract = replace(ESPN_FITNESS_CONTRACT, contract_version="test-contract-v2")
    policies = [ModularManagerPolicyNetwork() for _ in range(2)]
    scenario_bank = prepare_cuda_scenario_bank(
        [state],
        scenario_repeats=1,
        projection_noise=0.0,
        seed=13,
    )

    exact = evaluate_cuda_policy_population(
        policies,
        [state],
        scenario_bank=scenario_bank,
        enable_transactions=False,
        fitness_contract=contract,
        exact_policy_head_parity=True,
    )
    batched = evaluate_cuda_policy_population(
        policies,
        [state],
        scenario_bank=scenario_bank,
        enable_transactions=False,
        fitness_contract=contract,
        exact_policy_head_parity=False,
    )

    for expected, actual in zip(exact, batched, strict=True):
        assert actual.fitness == expected.fitness
        assert actual.wins == expected.wins
        assert actual.points_for == expected.points_for
