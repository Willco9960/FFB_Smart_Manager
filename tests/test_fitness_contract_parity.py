import torch

from fantasy_engine.lineup import ESPN_DEFAULT_LINEUP_RULES, score_starting_lineup
from fantasy_engine.player import Player
from gpu_sim.tensorized_draft import score_batched_lineups


def test_cpu_and_tensor_lineup_scores_match_full_espn_contract():
    roster = [
        Player("QB", "QB", "T", actual_score=20.0),
        Player("RB1", "RB", "T", actual_score=18.0),
        Player("RB2", "RB", "T", actual_score=16.0),
        Player("RB3", "RB", "T", actual_score=10.0),
        Player("WR1", "WR", "T", actual_score=19.0),
        Player("WR2", "WR", "T", actual_score=17.0),
        Player("TE", "TE", "T", actual_score=15.0),
        Player("DST", "DST", "T", actual_score=12.0),
        Player("K", "K", "T", actual_score=9.0),
    ]
    cpu_score = score_starting_lineup(roster, ESPN_DEFAULT_LINEUP_RULES)
    values = torch.tensor([[player.actual_score for player in roster]], dtype=torch.float32)
    positions = torch.tensor([0, 1, 1, 1, 2, 2, 3, 4, 5], dtype=torch.long)
    roster_indices = torch.tensor([list(range(len(roster)))], dtype=torch.long).unsqueeze(0)
    gpu_score = score_batched_lineups(
        values,
        values,
        positions,
        roster_indices,
        lineup_position_rules=((0,), (1,), (1,), (2,), (2,), (3,), (1, 2, 3), (4,), (5,)),
    ).scores.item()
    assert gpu_score == cpu_score
