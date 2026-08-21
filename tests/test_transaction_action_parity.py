from dataclasses import replace

import pytest
import torch

from fantasy_engine.transaction_contract import (
    TransactionEvent,
    canonical_player_key,
    canonical_waiver_action_key,
)
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
    assert state.rosters[0, 0].tolist() == [1, 4]
    assert state.rosters[0, 1].tolist() == [3, 0]


def test_cuda_trade_action_uses_shared_canonical_slot_tie_break():
    state = _minimal_state()
    state.weekly_projections[:, 0] = 10.0

    counts = state.apply_trades(0, top_k=2, minimum_improvement=0.0)

    assert counts.tolist() == [1]
    assert state.rosters[0].tolist() == [[2, 1], [0, 3]]


def test_cuda_replays_canonical_cpu_waiver_event_sequentially():
    state = _minimal_state()
    state.available[0, 4] = True
    event = TransactionEvent(
        season=2021,
        week=1,
        sequence_index=0,
        decision_type="waiver",
        team_name="team-0",
        action_key=canonical_waiver_action_key("team-0", "4", "0"),
        pre_state_digest=state._transaction_state_digest(0),
        post_state_digest="",
    )
    state.rosters[0, 0] = torch.tensor([1, 4])
    state.available[0, 4] = False
    state.available[0, 0] = True
    event = replace(event, post_state_digest=state._transaction_state_digest(0))
    state.rosters[0, 0] = torch.tensor([0, 1])
    state.available[0, 4] = True
    state.available[0, 0] = False

    state.replay_transaction_events([event])

    assert state.rosters[0, 0].tolist() == [1, 4]
    assert state.available[0, 4].item() is False
    assert state.available[0, 0].item() is True


def test_cuda_replay_rejects_pre_state_divergence():
    state = _minimal_state()
    event = TransactionEvent(
        season=2021,
        week=1,
        sequence_index=0,
        decision_type="waiver",
        team_name="team-0",
        action_key=canonical_waiver_action_key("team-0", "4", "0"),
        pre_state_digest="wrong",
        post_state_digest="wrong",
        accepted=False,
        rejection_reason="unavailable",
    )

    with pytest.raises(ValueError, match="pre-state divergence"):
        state.replay_transaction_events([event])


def test_cuda_replay_uses_full_cpu_player_identity_keys():
    state = _minimal_state()
    state.player_identity_keys = (
        canonical_player_key("p0", "WR", "ATL"),
        canonical_player_key("p1", "RB", "ATL"),
        canonical_player_key("p2", "QB", "BUF"),
        canonical_player_key("p3", "TE", "BUF"),
        canonical_player_key("p4", "WR", "DAL"),
        canonical_player_key("p5", "RB", "DAL"),
    )
    state.available[0, 4] = True
    event = TransactionEvent(
        season=2021,
        week=1,
        sequence_index=0,
        decision_type="waiver",
        team_name="Team 1",
        action_key="waiver|Team 1|p4|WR|DAL|p0|WR|ATL",
        pre_state_digest=state._transaction_state_digest(0),
        post_state_digest="",
        player_keys=(state.player_identity_keys[4], state.player_identity_keys[0]),
        team_index=0,
    )
    state.rosters[0, 0] = torch.tensor([1, 4])
    state.available[0, 4] = False
    state.available[0, 0] = True
    event = replace(event, post_state_digest=state._transaction_state_digest(0))
    state.rosters[0, 0] = torch.tensor([0, 1])
    state.available[0, 4] = True
    state.available[0, 0] = False

    state.replay_transaction_events([event])

    assert state.rosters[0, 0].tolist() == [1, 4]
