import torch

from gpu_sim.full_season import create_synthetic_season_state


def _minimal_state():
    state = create_synthetic_season_state(
        scenarios=1,
        players=6,
        team_count=2,
        roster_size=2,
        weeks=17,
    )
    state.lineup_position_rules = ((0,),)
    state.positions.zero_()
    state.rosters = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
    state.available = torch.zeros((1, 6), dtype=torch.bool)
    return state


def test_cuda_waiver_action_uses_shared_canonical_player_tie_break():
    state = _minimal_state()
    state.available[0, 4:] = True
    state.weekly_projections[:, 0] = 10.0

    counts = state.apply_waivers(0, minimum_improvement=0.0)

    assert counts.tolist() == [2]
    assert state.rosters[0, 0].tolist() == [4, 1]
    assert state.rosters[0, 1].tolist() == [0, 3]


def test_cuda_trade_action_uses_shared_canonical_slot_tie_break():
    state = _minimal_state()
    state.weekly_projections[:, 0] = 10.0

    counts = state.apply_trades(0, top_k=2, minimum_improvement=0.0)

    assert counts.tolist() == [1]
    assert state.rosters[0].tolist() == [[2, 1], [0, 3]]
