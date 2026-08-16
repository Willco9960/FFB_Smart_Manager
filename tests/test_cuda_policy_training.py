import torch

from gpu_sim.full_season import create_synthetic_season_state, run_full_cuda_season
from gpu_sim.policy_training import evaluate_cuda_policy, train_cuda_policy_population
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
