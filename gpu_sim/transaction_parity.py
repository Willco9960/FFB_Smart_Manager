"""Deterministic transaction fixtures shared by CPU/CUDA parity tests.

The production agents are allowed to choose different actions.  These
fixtures instead apply the *same* legal waiver/trade actions to both backends,
so roster mutation, legality, and bookkeeping can be tested independently of
policy quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gpu_sim.full_season import CudaSeasonState


@dataclass(frozen=True)
class WaiverFixture:
    week: int
    team_index: int
    add_player_index: int
    drop_player_index: int


@dataclass(frozen=True)
class TradeFixture:
    week: int
    proposing_team_index: int
    receiving_team_index: int
    offered_player_index: int
    requested_player_index: int


@dataclass(frozen=True)
class TransactionFixtureResult:
    waiver_count: int
    trade_count: int
    roster_snapshot: tuple[tuple[tuple[int, ...], ...], ...]


def _find_roster_slot(rosters: torch.Tensor, team_index: int, player_index: int) -> torch.Tensor:
    matches = rosters[:, team_index, :] == player_index
    if not bool(matches.any(dim=1).all().item()):
        raise ValueError(f"Player {player_index} is not on team {team_index}.")
    return matches.to(torch.int64).argmax(dim=1)


def apply_fixture_to_cuda(
    state: CudaSeasonState,
    *,
    waivers: tuple[WaiverFixture, ...] = (),
    trades: tuple[TradeFixture, ...] = (),
) -> TransactionFixtureResult:
    """Apply an explicit legal action sequence to every CUDA scenario."""

    rosters = state.rosters
    available = state.available
    if rosters is None or available is None:
        raise ValueError("CUDA state must be initialized before applying fixtures.")

    waiver_count = 0
    for action in waivers:
        if not bool(available[:, action.add_player_index].all().item()):
            raise ValueError("Fixture waiver add player is not available.")
        drop_slot = _find_roster_slot(rosters, action.team_index, action.drop_player_index)
        rosters[:, action.team_index, drop_slot] = action.add_player_index
        available[:, action.add_player_index] = False
        available[:, action.drop_player_index] = True
        waiver_count += 1

    trade_count = 0
    for action in trades:
        offered_slot = _find_roster_slot(
            rosters, action.proposing_team_index, action.offered_player_index
        )
        requested_slot = _find_roster_slot(
            rosters, action.receiving_team_index, action.requested_player_index
        )
        rosters[:, action.proposing_team_index, offered_slot] = action.requested_player_index
        rosters[:, action.receiving_team_index, requested_slot] = action.offered_player_index
        trade_count += 1

    snapshot = tuple(
        tuple(tuple(int(player) for player in roster) for roster in scenario)
        for scenario in rosters.detach().cpu().tolist()
    )
    return TransactionFixtureResult(waiver_count, trade_count, snapshot)
