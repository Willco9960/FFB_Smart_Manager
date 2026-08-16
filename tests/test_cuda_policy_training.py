import random

import pytest
import torch

from gpu_sim.full_season import create_synthetic_season_state, run_full_cuda_season
from gpu_sim.policy_training import (
    evaluate_cuda_policy,
    evaluate_cuda_policy_population,
    mutate_policy,
    prepare_cuda_scenario_bank,
    save_cuda_training_state,
    train_cuda_policy_population,
    validate_cuda_training_state_contract,
)
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

    def checkpoint_callback(generation, population, best_policy, metrics, rng):
        save_cuda_training_state(
            checkpoint_path,
            generation=generation,
            population=population,
            best_policy=best_policy,
            metrics=metrics,
            rng_state=rng.getstate(),
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
    )
    resume_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    )

    assert len(first_metrics) == 1
    assert len(resumed_metrics) == 2
    assert resumed_metrics[0].generation == 1
    assert resumed_metrics[1].generation == 2


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
