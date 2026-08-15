import pytest
import torch

from gpu_sim.tensorized_draft import (
    benchmark_tensorized_for_duration,
    run_batched_greedy_draft,
    run_cpu_reference_greedy_draft,
    score_batched_lineups,
    score_batched_offensive_lineups,
)


def _fixture():
    projected = torch.tensor(
        [[100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]],
        dtype=torch.float32,
    )
    actual = projected + 1.0
    positions = torch.tensor([0, 1, 1, 2, 2, 3, 1, 2, 3, 1], dtype=torch.long)
    return projected, actual, positions


def test_tensorized_draft_matches_cpu_reference():
    projected, _, _ = _fixture()
    expected = run_cpu_reference_greedy_draft(projected, team_count=2, rounds=3)
    actual = run_batched_greedy_draft(projected, team_count=2, rounds=3)

    assert torch.equal(actual.player_indices.cpu(), expected.player_indices.cpu())


def test_tensorized_draft_never_selects_a_player_twice():
    projected, _, _ = _fixture()
    result = run_batched_greedy_draft(projected, team_count=2, rounds=3)

    assert len(result.player_indices.unique()) == 6


def test_offensive_lineup_scoring_uses_best_legal_slots():
    projected, actual, positions = _fixture()
    draft = run_batched_greedy_draft(projected, team_count=1, rounds=8)
    lineup = score_batched_offensive_lineups(actual, positions, draft.player_indices)

    assert lineup.scores.shape == (1, 1)
    assert lineup.scores.item() > 0
    assert lineup.player_indices.shape == (1, 1, 7)


def test_lineup_selection_can_use_projection_and_score_actual_points():
    projected, actual, positions = _fixture()
    roster = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=torch.long)
    lineup = score_batched_lineups(projected, actual, positions, roster)
    expected = actual[0, [0, 1, 2, 3, 4, 5, 6]].sum()

    assert torch.isclose(lineup.scores[0, 0], expected)


def test_incomplete_lineup_gets_zero_score():
    actual = torch.tensor([[10.0, 9.0, 8.0]], dtype=torch.float32)
    positions = torch.tensor([0, 1, 2], dtype=torch.long)
    rosters = torch.tensor([[[0, 1, 2]]], dtype=torch.long)

    lineup = score_batched_offensive_lineups(actual, positions, rosters)

    assert lineup.scores.item() == 0.0


def test_duration_benchmark_completes_bounded_smoke_run():
    projected, actual, positions = _fixture()
    result = benchmark_tensorized_for_duration(
        projected,
        actual,
        positions,
        team_count=1,
        rounds=8,
        duration_seconds=0.01,
        progress_seconds=0.01,
    )

    assert result["completed_batches"] > 0
    assert result["elapsed_seconds"] >= 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_matches_cpu_reference():
    projected, actual, positions = _fixture()
    cpu_draft = run_batched_greedy_draft(projected, team_count=1, rounds=8)
    cuda_draft = run_batched_greedy_draft(
        projected.cuda(), team_count=1, rounds=8
    ).player_indices.cpu()
    assert torch.equal(cuda_draft, cpu_draft.player_indices.cpu())

    cpu_score = score_batched_offensive_lineups(
        actual,
        positions,
        cpu_draft.player_indices,
    ).scores
    cuda_score = score_batched_offensive_lineups(
        actual.cuda(),
        positions.cuda(),
        cuda_draft.cuda(),
    ).scores.cpu()
    assert torch.allclose(cuda_score, cpu_score)
