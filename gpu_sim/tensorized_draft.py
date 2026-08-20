"""Reference and experimental tensorized draft/lineup kernels.

This is deliberately a *bounded* CUDA experiment. It models the deterministic
projection-best draft and offensive lineup score so we can measure the upper
bound of moving dense state transitions to a GPU. Waivers, trades, playoffs,
and the full object-oriented league engine remain CPU-owned until parity is
proven separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from gpu_sim.profiling import TensorStageProfiler


@dataclass(frozen=True)
class DraftBatchResult:
    """Drafted player indices for every scenario and team."""

    player_indices: torch.Tensor


@dataclass(frozen=True)
class LineupBatchResult:
    """Starting-lineup indices and scores for every scenario and team."""

    player_indices: torch.Tensor
    scores: torch.Tensor


def _validate_draft_inputs(
    projected_points: torch.Tensor,
    team_count: int,
    rounds: int,
) -> None:
    if projected_points.ndim != 2:
        raise ValueError("projected_points must have shape [scenarios, players].")
    if team_count < 1 or rounds < 1:
        raise ValueError("team_count and rounds must be positive.")
    if projected_points.shape[1] < team_count * rounds:
        raise ValueError("There must be at least one player per draft pick.")


def run_batched_greedy_draft(
    projected_points: torch.Tensor,
    team_count: int = 10,
    rounds: int = 16,
) -> DraftBatchResult:
    """Run projection-best snake drafts for a batch of scenarios.

    The implementation keeps the draft state in tensors and performs one
    batched ``argmax`` per pick. This matches the CPU fallback's default
    projection-best policy, while making scenario parallelism explicit.
    """

    _validate_draft_inputs(projected_points, team_count, rounds)
    scenarios, player_count = projected_points.shape
    working_scores = projected_points.clone()
    rosters = torch.empty(
        (scenarios, team_count, rounds),
        dtype=torch.long,
        device=projected_points.device,
    )

    pick_index = 0
    for round_number in range(rounds):
        order = range(team_count) if round_number % 2 == 0 else range(team_count - 1, -1, -1)
        for team_index in order:
            selected = working_scores.argmax(dim=1)
            rosters[:, team_index, pick_index // team_count] = selected
            working_scores.scatter_(1, selected.unsqueeze(1), float("-inf"))
            pick_index += 1

    return DraftBatchResult(player_indices=rosters)


def run_batched_roster_aware_draft(
    projected_points: torch.Tensor,
    positions: torch.Tensor,
    team_count: int = 10,
    rounds: int = 16,
    position_minimums: tuple[int, ...] = (1, 4, 4, 1),
    position_maximums: tuple[int, ...] = (2, 6, 7, 3),
) -> DraftBatchResult:
    """Draft projection-best rosters while preserving CPU roster bounds.

    Position IDs are QB=0, RB=1, WR=2, TE=3. The constraint check mirrors the
    CPU genome drafter's per-team minimum/maximum shape for the offensive
    roster. It intentionally leaves transaction and policy scoring separate.
    """

    _validate_draft_inputs(projected_points, team_count, rounds)
    if positions.shape != (projected_points.shape[1],):
        raise ValueError("positions must describe every player.")
    if len(position_minimums) != len(position_maximums):
        raise ValueError("Position minimums and maximums must have equal length.")
    if sum(position_minimums) > rounds:
        raise ValueError("Position minimums cannot exceed roster rounds.")
    if any(
        minimum < 0 or maximum < minimum
        for minimum, maximum in zip(position_minimums, position_maximums, strict=True)
    ):
        raise ValueError("Position maximums must be at least their minimums.")
    scenarios, player_count = projected_points.shape
    position_count = len(position_minimums)
    if positions.max().item() >= position_count:
        raise ValueError("positions contain an unsupported position ID.")

    working_scores = projected_points.clone()
    rosters = torch.full(
        (scenarios, team_count, rounds),
        -1,
        dtype=torch.long,
        device=projected_points.device,
    )
    position_one_hot = torch.nn.functional.one_hot(
        positions,
        num_classes=position_count,
    ).to(torch.int16)
    maximums = torch.tensor(position_maximums, device=projected_points.device)
    counts = torch.zeros(
        (scenarios, team_count, position_count),
        dtype=torch.int16,
        device=projected_points.device,
    )
    next_slots = torch.zeros(team_count, dtype=torch.long, device=projected_points.device)

    def select_player(allowed: torch.Tensor, team_indices: torch.Tensor) -> None:
        candidates = working_scores.masked_fill(~allowed, float("-inf"))
        if not torch.isfinite(candidates.max(dim=1).values).all():
            # A small/filtered fixture may not contain enough players to satisfy
            # every global minimum. Preserve the historical simulator's safe
            # fallback instead of crashing or selecting an already-used player.
            candidates = working_scores
        if not torch.isfinite(candidates.max(dim=1).values).all():
            raise ValueError("No remaining finite player is available for the roster.")
        selected = candidates.argmax(dim=1)
        rosters[:, team_indices, next_slots[team_indices]] = selected.unsqueeze(1)
        counts[:, team_indices] += position_one_hot[selected].unsqueeze(1)
        working_scores.scatter_(1, selected.unsqueeze(1), float("-inf"))
        next_slots[team_indices] += 1

    # Reserve every team's mandatory position minimum before filling extras.
    # This prevents scarce TE/K/DST players from being consumed by earlier
    # projection-best teams and leaving an illegal baseline roster.
    for position, minimum in enumerate(position_minimums):
        eligible = (positions == position).unsqueeze(0).expand(scenarios, -1)
        for required_round in range(minimum):
            order = (
                range(team_count)
                if required_round % 2 == 0
                else range(team_count - 1, -1, -1)
            )
            for team_index in order:
                select_player(eligible, torch.tensor([team_index], device=projected_points.device))

    for round_number in range(sum(position_minimums), rounds):
        order = range(team_count) if round_number % 2 == 0 else range(team_count - 1, -1, -1)
        for team_index in order:
            team_counts = counts[:, team_index]
            candidate_counts = team_counts.unsqueeze(1) + position_one_hot.unsqueeze(0)
            allowed = (candidate_counts <= maximums).all(dim=2)
            allowed &= torch.isfinite(working_scores)
            select_player(allowed, torch.tensor([team_index], device=projected_points.device))

    return DraftBatchResult(player_indices=rosters)


def run_cpu_reference_greedy_draft(
    projected_points: torch.Tensor,
    team_count: int = 10,
    rounds: int = 16,
) -> DraftBatchResult:
    """Small, readable CPU reference used for parity tests and benchmarks."""

    _validate_draft_inputs(projected_points, team_count, rounds)
    values = projected_points.detach().cpu().tolist()
    rosters: list[list[list[int]]] = []

    for scenario_values in values:
        available = set(range(len(scenario_values)))
        scenario_rosters = [[-1] * rounds for _ in range(team_count)]
        for round_number in range(rounds):
            order = range(team_count) if round_number % 2 == 0 else range(team_count - 1, -1, -1)
            for team_index in order:
                selected = max(available, key=scenario_values.__getitem__)
                scenario_rosters[team_index][round_number] = selected
                available.remove(selected)
        rosters.append(scenario_rosters)

    return DraftBatchResult(
        player_indices=torch.tensor(rosters, dtype=torch.long, device=projected_points.device)
    )


def _select_best_slot(
    selection_scores: torch.Tensor,
    actual_scores: torch.Tensor,
    positions: torch.Tensor,
    used: torch.Tensor,
    eligible_positions: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    eligible = torch.zeros_like(positions, dtype=torch.bool)
    for position in eligible_positions:
        eligible |= positions == position
    candidates = selection_scores.masked_fill(~eligible | used, float("-inf"))
    selected_scores, selected_indices = candidates.max(dim=2)
    found = torch.isfinite(selected_scores)
    selected_actual_scores = actual_scores.gather(2, selected_indices.unsqueeze(2)).squeeze(2)
    selected_actual_scores = torch.where(
        found,
        selected_actual_scores,
        torch.full_like(selected_actual_scores, float("-inf")),
    )
    used.scatter_(2, selected_indices.unsqueeze(2), found.unsqueeze(2))
    return selected_indices, selected_actual_scores


def score_batched_lineups(
    selection_points: torch.Tensor,
    actual_points: torch.Tensor,
    positions: torch.Tensor,
    rosters: torch.Tensor,
    lineup_position_rules: tuple[tuple[int, ...], ...] = (
        (0,), (1,), (1,), (2,), (2,), (3,), (1, 2, 3)
    ),
) -> LineupBatchResult:
    """Select legal offensive lineups by projections and score actual points."""

    # Vectorized team path: flatten [scenario, team, player] into the existing
    # exact scorer.  This preserves the per-team roster/slot semantics while
    # eliminating one Python/kernel launch sequence per team.
    if selection_points.ndim == 3:
        if actual_points.ndim != 2 or rosters.ndim != 3:
            raise ValueError(
                "Team-batched points require [scenarios, players] actuals and rosters."
            )
        scenarios, teams, players = selection_points.shape
        if actual_points.shape != (scenarios, players):
            raise ValueError("Team-batched actual points must match scenario/player dimensions.")
        if rosters.shape[0:2] != (scenarios, teams):
            raise ValueError("Team-batched rosters must match scenario/team dimensions.")
        flattened = score_batched_lineups(
            selection_points.reshape(scenarios * teams, players),
            actual_points.unsqueeze(1)
            .expand(scenarios, teams, players)
            .reshape(scenarios * teams, players),
            positions,
            rosters.reshape(scenarios * teams, rosters.shape[2]).unsqueeze(1),
            lineup_position_rules,
        )
        return LineupBatchResult(
            player_indices=flattened.player_indices.reshape(scenarios, teams, -1),
            scores=flattened.scores.reshape(scenarios, teams),
        )

    if selection_points.ndim != 2 or actual_points.ndim != 2:
        raise ValueError("Point tensors must have shape [scenarios, players].")
    if selection_points.shape != actual_points.shape:
        raise ValueError("Selection and actual points must have the same shape.")
    if positions.ndim != 1 or rosters.ndim != 3:
        raise ValueError("Invalid tensor dimensions for lineup scoring.")
    if selection_points.shape[1] != positions.shape[0]:
        raise ValueError("positions must describe every player.")
    if rosters.shape[0] != selection_points.shape[0]:
        raise ValueError("rosters and point tensors must share scenario count.")

    roster_selection_scores = selection_points.gather(
        1,
        rosters.reshape(rosters.shape[0], -1),
    ).reshape(*rosters.shape)
    roster_actual_scores = actual_points.gather(
        1,
        rosters.reshape(rosters.shape[0], -1),
    ).reshape(*rosters.shape)
    roster_positions = positions[rosters]
    used = torch.zeros_like(roster_selection_scores, dtype=torch.bool)
    selected_indices: list[torch.Tensor] = []
    selected_scores: list[torch.Tensor] = []

    for eligible_positions in lineup_position_rules:
        indices, scores = _select_best_slot(
            roster_selection_scores,
            roster_actual_scores,
            roster_positions,
            used,
            eligible_positions,
        )
        selected_indices.append(indices)
        selected_scores.append(scores)

    selected_score_tensor = torch.stack(selected_scores, dim=2)
    complete = torch.isfinite(selected_score_tensor).all(dim=2)
    total = selected_score_tensor.masked_fill(~torch.isfinite(selected_score_tensor), 0.0).sum(
        dim=2
    )
    total = torch.where(complete, total, torch.zeros_like(total))
    return LineupBatchResult(
        player_indices=torch.stack(selected_indices, dim=2),
        scores=total,
    )


def score_batched_offensive_lineups(
    actual_points: torch.Tensor,
    positions: torch.Tensor,
    rosters: torch.Tensor,
) -> LineupBatchResult:
    """Score QB/2RB/2WR/TE/FLEX lineups for batched drafted rosters.

    ``actual_points`` has shape ``[scenarios, players]``; ``positions`` has
    shape ``[players]`` with IDs 0=QB, 1=RB, 2=WR, 3=TE; and ``rosters`` has
    shape ``[scenarios, teams, roster_slots]``. Incomplete lineups receive a
    score of zero, matching the training engine's invalid-roster penalty.
    """

    return score_batched_lineups(
        selection_points=actual_points,
        actual_points=actual_points,
        positions=positions,
        rosters=rosters,
    )


def benchmark_tensorized_draft(
    projected_points: torch.Tensor,
    actual_points: torch.Tensor,
    positions: torch.Tensor,
    team_count: int = 10,
    rounds: int = 16,
    repeats: int = 3,
    profile_stages: bool = False,
) -> dict[str, float | int | str]:
    """Benchmark the tensorized draft and lineup pipeline on one device."""

    if repeats < 1:
        raise ValueError("repeats must be positive.")
    device = projected_points.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    result = None
    profiler = TensorStageProfiler(device) if profile_stages else None
    for _ in range(repeats):
        if profiler is None:
            result = run_batched_greedy_draft(projected_points, team_count, rounds)
            score_batched_offensive_lineups(actual_points, positions, result.player_indices)
        else:
            with profiler.stage("draft"):
                result = run_batched_greedy_draft(projected_points, team_count, rounds)
            with profiler.stage("lineup"):
                score_batched_offensive_lineups(
                    actual_points,
                    positions,
                    result.player_indices,
                )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    batch_runs_per_hour = repeats / (elapsed / 3600.0)
    scenario_runs_per_hour = projected_points.shape[0] * batch_runs_per_hour
    assert result is not None
    return_value: dict[str, object] = {
        "device": str(device),
        "scenarios": projected_points.shape[0],
        "players": projected_points.shape[1],
        "teams": team_count,
        "rounds": rounds,
        "repeats": repeats,
        "elapsed_seconds": round(elapsed, 6),
        # ``generations_per_hour`` is retained as a compatibility alias for
        # the benchmark's batch-level rate. It is not full-season trainer GPH.
        "generations_per_hour": round(batch_runs_per_hour, 2),
        "batch_runs_per_hour": round(batch_runs_per_hour, 2),
        "scenario_runs_per_hour": round(scenario_runs_per_hour, 2),
    }
    if profiler is not None:
        return_value["stages"] = profiler.as_dict()
    return return_value


def benchmark_tensorized_for_duration(
    projected_points: torch.Tensor,
    actual_points: torch.Tensor,
    positions: torch.Tensor,
    team_count: int = 10,
    rounds: int = 16,
    duration_seconds: float = 3600.0,
    progress_seconds: float = 30.0,
    progress_callback=None,
) -> dict[str, float | int | str]:
    """Run the same tensorized workload continuously for a bounded duration."""

    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive.")
    if progress_seconds <= 0.0:
        raise ValueError("progress_seconds must be positive.")

    device = projected_points.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    next_progress = progress_seconds
    completed_batches = 0

    while True:
        result = run_batched_greedy_draft(projected_points, team_count, rounds)
        score_batched_offensive_lineups(actual_points, positions, result.player_indices)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - start
        completed_batches += 1

        if progress_callback is not None and elapsed >= next_progress:
            progress_callback(completed_batches, elapsed)
            next_progress += progress_seconds
        if elapsed >= duration_seconds:
            break

    elapsed = perf_counter() - start
    batch_runs_per_hour = completed_batches / (elapsed / 3600.0)
    scenario_runs_per_hour = projected_points.shape[0] * batch_runs_per_hour
    return {
        "device": str(device),
        "scenarios": projected_points.shape[0],
        "players": projected_points.shape[1],
        "teams": team_count,
        "rounds": rounds,
        "duration_seconds_requested": duration_seconds,
        "elapsed_seconds": round(elapsed, 6),
        "completed_batches": completed_batches,
        "generations_per_hour": round(batch_runs_per_hour, 2),
        "batch_runs_per_hour": round(batch_runs_per_hour, 2),
        "scenario_runs_per_hour": round(scenario_runs_per_hour, 2),
    }
