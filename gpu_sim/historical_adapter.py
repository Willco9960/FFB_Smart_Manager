"""Leakage-safe conversion of historical seasons into CUDA tensors.

The adapter mirrors the current CPU projection contract: a season's draft
projection comes from the previous season, and weekly projections use only
performances from weeks before the decision week. It intentionally targets the
CPU pipeline's offensive lineup mode while special-teams parity is still
tracked separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from fantasy_engine.historical_loader import RAW_DATA_DIR, load_player_stats
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.player import Player
from fantasy_engine.weekly_data import WeeklyPlayerPerformance, load_weekly_performances
from fantasy_engine.weekly_projection import calculate_weekly_projection
from gpu_sim.full_season import CudaSeasonState


@dataclass(frozen=True)
class HistoricalCudaInputs:
    """CPU objects and tensor state for one matched historical season."""

    season: int
    projection_season: int
    players: tuple[Player, ...]
    performances: tuple[WeeklyPlayerPerformance, ...]
    state: CudaSeasonState


def _performance_history(
    performances: list[WeeklyPlayerPerformance],
) -> dict[tuple[str, str], list[WeeklyPlayerPerformance]]:
    history: dict[tuple[str, str], list[WeeklyPlayerPerformance]] = {}
    for performance in performances:
        history.setdefault((performance.player_name, performance.position), []).append(performance)
    return history


def create_historical_cuda_inputs(
    season: int,
    *,
    projection_season: int | None = None,
    players: int = 256,
    weeks: int = 17,
    raw_data_dir: Path = RAW_DATA_DIR,
    device: torch.device | str = "cpu",
) -> HistoricalCudaInputs:
    """Load one season using the same cutoff rules as the CPU simulator."""

    if season < 1:
        raise ValueError("season must be positive.")
    if players < 160:
        raise ValueError("players must hold ten 16-player ESPN rosters.")
    if weeks < 17:
        raise ValueError("weeks must include the three playoff weeks.")

    projection_season = season - 1 if projection_season is None else projection_season
    # Validate both files before constructing the tensor state. This keeps a
    # missing historical input from producing a silently zero-filled season.
    load_player_stats(season=projection_season, raw_data_dir=raw_data_dir)
    load_player_stats(season=season, raw_data_dir=raw_data_dir)
    player_pool = load_leakage_safe_player_pool(
        projection_season=projection_season,
        actual_season=season,
        raw_data_dir=raw_data_dir,
    )[:players]
    if len(player_pool) < 160:
        raise ValueError(
            f"Season {season} has only {len(player_pool)} leakage-safe players; "
            "at least 160 are required."
        )

    performances = load_weekly_performances(
        season=season,
        raw_data_dir=raw_data_dir,
    )
    history = _performance_history(performances)
    player_keys = {(player.name, player.position) for player in player_pool}
    actual_by_week = {
        (
            performance.week,
            performance.player_name,
            performance.position,
        ): performance.fantasy_points
        for performance in performances
        if (performance.player_name, performance.position) in player_keys
    }

    draft_projections = torch.tensor(
        [[player.projected_score for player in player_pool]],
        dtype=torch.float32,
    )
    weekly_projections = torch.zeros((1, weeks, len(player_pool)), dtype=torch.float32)
    weekly_actual_points = torch.zeros_like(weekly_projections)
    for player_index, player in enumerate(player_pool):
        player_history = history.get((player.name, player.position), [])
        for week_index in range(weeks):
            historical_week = week_index + 1
            weekly_projections[0, week_index, player_index] = calculate_weekly_projection(
                player,
                [item for item in player_history if item.week < historical_week],
                season_length=14,
            )
            weekly_actual_points[0, week_index, player_index] = actual_by_week.get(
                (historical_week, player.name, player.position),
                0.0,
            )

    position_ids = torch.tensor(
        [{"QB": 0, "RB": 1, "WR": 2, "TE": 3}[player.position] for player in player_pool],
        dtype=torch.long,
    )
    state = CudaSeasonState(
        draft_projections=draft_projections.to(device),
        weekly_projections=weekly_projections.to(device),
        weekly_actual_points=weekly_actual_points.to(device),
        positions=position_ids.to(device),
    )
    return HistoricalCudaInputs(
        season=season,
        projection_season=projection_season,
        players=tuple(player_pool),
        performances=tuple(performances),
        state=state,
    )
