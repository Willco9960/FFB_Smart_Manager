import torch

from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from fantasy_engine.transactions import TradeProposal, WaiverClaim, apply_trade, apply_waiver_claim
from gpu_sim.full_season import CudaSeasonState
from gpu_sim.transaction_parity import (
    TradeFixture,
    WaiverFixture,
    apply_fixture_to_cuda,
)


def test_explicit_transaction_fixture_matches_cpu_roster_mutation():
    players = [Player(f"P{index}", "RB", "T", player_id=f"id-{index}") for index in range(8)]
    cpu = League(
        name="fixture",
        teams=[Team("Team 1", [players[0], players[1]]), Team("Team 2", [players[2], players[3]])],
        available_players=players[4:],
    )
    waiver = WaiverClaim("Team 1", players[4], players[1], week=1)
    apply_waiver_claim(cpu, waiver)
    trade = TradeProposal(
        "Team 1",
        "Team 2",
        (players[0],),
        (players[2],),
        week=2,
    )
    apply_trade(cpu, trade)

    state = CudaSeasonState(
        draft_projections=torch.ones((1, 8)),
        weekly_projections=torch.ones((1, 17, 8)),
        weekly_actual_points=torch.ones((1, 17, 8)),
        positions=torch.ones(8, dtype=torch.long),
        team_count=2,
        roster_size=2,
        rosters=torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long),
        available=torch.tensor([[False, False, False, False, True, True, True, True]]),
    )
    result = apply_fixture_to_cuda(
        state,
        waivers=(WaiverFixture(1, 0, 4, 1),),
        trades=(TradeFixture(2, 0, 1, 0, 2),),
    )

    cpu_snapshot = tuple(
        tuple(sorted(player.player_id for player in team.roster)) for team in cpu.teams
    )
    cuda_snapshot = tuple(
        tuple(sorted(f"id-{index}" for index in roster))
        for roster in result.roster_snapshot[0]
    )
    assert cpu_snapshot == cuda_snapshot
    assert result.waiver_count == 1
    assert result.trade_count == 1
