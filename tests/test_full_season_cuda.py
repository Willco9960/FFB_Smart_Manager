import pytest
import torch

from fantasy_engine.fitness_contract import ESPN_FITNESS_CONTRACT
from gpu_sim.full_season import create_synthetic_season_state, run_full_cuda_season
from models.modular_manager_policy import ModularManagerPolicyNetwork


def test_cuda_season_draft_and_weekly_scoring_have_expected_shapes():
    state = create_synthetic_season_state(scenarios=2, players=200)
    draft = state.draft()
    scores = state.score_week(0)

    assert draft.player_indices.shape == (2, 10, 16)
    assert state.available.shape == (2, 200)
    assert scores.shape == (2, 10)
    assert state.wins.sum().item() == 10
    assert state.losses.sum().item() == 10
    assert state.ties.sum().item() == 0
    assert torch.all(state.points_against >= 0)


def test_cuda_season_waivers_update_rosters_and_counts():
    state = create_synthetic_season_state(scenarios=2, players=200)
    state.draft()
    free_agent = int(torch.where(state.available[0])[0][0].item())
    state.weekly_projections[:, 0, free_agent] = 1000.0
    counts = state.apply_waivers(0)

    assert counts.shape == (2,)
    assert counts.max().item() >= 1
    assert len(state.waiver_counts) == 1


def test_cuda_season_trades_return_per_scenario_counts():
    state = create_synthetic_season_state(scenarios=2, players=200)
    state.draft()
    counts = state.apply_trades(0)

    assert counts.shape == (2,)
    assert torch.all(counts >= 0)
    assert len(state.trade_counts) == 1


def test_cuda_season_playoffs_return_valid_champions():
    state = create_synthetic_season_state(scenarios=2, players=200)
    state.draft()
    for week in range(14):
        state.score_week(week)

    champions = state.run_playoffs()

    assert champions.shape == (2,)
    assert torch.all((champions >= 0) & (champions < 10))
    assert state.playoff_wins is not None


def test_full_cuda_season_pipeline_runs_end_to_end():
    state = create_synthetic_season_state(scenarios=1, players=200)
    result = run_full_cuda_season(state)

    assert len(result.weekly_scores) == 14
    assert len(result.waiver_counts) == 14
    assert len(result.trade_counts) == 14
    assert result.champions is not None


def test_policy_controlled_full_season_records_all_in_season_heads():
    state = create_synthetic_season_state(scenarios=1, players=200)

    run_full_cuda_season(
        state,
        policy_network=ModularManagerPolicyNetwork(),
    )

    assert len(state.lineup_policy_gains) == 14
    assert len(state.waiver_policy_gains) == 14
    assert len(state.trade_policy_gains) == 14


def test_full_contract_draft_reserves_kicker_and_defense_slots():
    generator = torch.Generator().manual_seed(7)
    positions = torch.tensor(([0, 1, 2, 3, 4, 5] * 40)[:200])
    state = create_synthetic_season_state(scenarios=1, players=200)
    state.positions = positions
    state.draft_projections = torch.rand((1, 200), generator=generator) * 100.0
    state.weekly_projections = torch.rand((1, 17, 200), generator=generator) * 20.0
    state.weekly_actual_points = torch.rand((1, 17, 200), generator=generator) * 20.0

    # The public full-season entrypoint expands contract slot counts to the
    # nine legal ESPN starter slots before scoring.
    run_full_cuda_season(state, enable_transactions=False, fitness_contract=ESPN_FITNESS_CONTRACT)
    roster_positions = state.positions[state.rosters[0]]
    assert (roster_positions == 4).sum(dim=1).min().item() >= 1
    assert (roster_positions == 5).sum(dim=1).min().item() >= 1
    assert len(state.lineup_position_rules) == 9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_full_cuda_season_pipeline_runs_on_cuda():
    state = create_synthetic_season_state(scenarios=1, players=200, device="cuda")
    result = run_full_cuda_season(state)

    assert result.device.type == "cuda"
    assert result.champions is not None
