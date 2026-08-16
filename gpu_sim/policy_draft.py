"""CUDA policy-conditioned draft kernels.

The production CPU simulator evaluates object-oriented agents.  This module
keeps the same leakage-safe draft features in tensors so a candidate policy can
draft against projection-best opponents without moving every pick through
Python objects.
"""

from __future__ import annotations

import torch

from gpu_sim.tensorized_draft import DraftBatchResult


def _validate_inputs(
    projected_points: torch.Tensor,
    positions: torch.Tensor,
    team_count: int,
    rounds: int,
) -> None:
    if projected_points.ndim != 2:
        raise ValueError("projected_points must have shape [scenarios, players].")
    if positions.shape != (projected_points.shape[1],):
        raise ValueError("positions must describe every player.")
    if team_count < 1 or rounds < 1:
        raise ValueError("team_count and rounds must be positive.")
    if projected_points.shape[1] < team_count * rounds:
        raise ValueError("There must be enough players to fill every roster.")
    if positions.numel() and (positions.min() < 0 or positions.max() > 3):
        raise ValueError("positions must use QB=0, RB=1, WR=2, TE=3.")


def _position_one_hot(positions: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(positions, num_classes=4).to(torch.float32)


def _build_player_features(
    projected_points: torch.Tensor,
    positions: torch.Tensor,
    available: torch.Tensor,
    rosters: torch.Tensor,
    team_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the 14+25 feature tensors used by ModularManagerPolicyNetwork."""

    scenarios, player_count = projected_points.shape
    roster = rosters[:, team_index]
    roster_positions = positions[roster]
    roster_counts = torch.stack(
        [(roster_positions == position).sum(dim=1) for position in range(4)],
        dim=1,
    ).to(torch.float32)
    roster_size = (roster >= 0).sum(dim=1).to(torch.float32)
    roster_points = projected_points.gather(1, roster.clamp_min(0)).masked_fill(
        roster < 0, 0.0
    ).sum(dim=1)

    available_count = available.sum(dim=1).to(torch.float32)
    available_one_hot = _position_one_hot(positions).unsqueeze(0)
    available_by_position = available.unsqueeze(2) & (available_one_hot > 0)
    available_position_counts = available_by_position.sum(dim=1).to(torch.float32)
    highest_available = projected_points.masked_fill(~available, float("-inf")).max(dim=1).values
    highest_available = highest_available.clamp_min(1.0)
    candidate_same_position = available_position_counts.gather(
        1, positions.unsqueeze(0).expand(scenarios, -1)
    )
    # Rank each position with argsort/scatter rather than a dense
    # [scenarios, players, players] comparison.  The old pairwise comparison
    # was quadratic in the player pool and dominated repeated draft picks.
    greater_same_position = torch.zeros(
        (scenarios, player_count), dtype=torch.float32, device=projected_points.device
    )
    for position in range(4):
        position_mask = positions == position
        position_values = projected_points[:, position_mask].masked_fill(
            ~available[:, position_mask], float("-inf")
        )
        if position_values.shape[1] == 0:
            continue
        order = torch.argsort(position_values, dim=1, descending=True)
        sorted_ranks = torch.arange(
            1,
            position_values.shape[1] + 1,
            device=projected_points.device,
            dtype=torch.float32,
        ).expand(scenarios, -1)
        original_ranks = torch.empty_like(sorted_ranks)
        original_ranks.scatter_(1, order, sorted_ranks)
        greater_same_position[:, position_mask] = original_ranks
    position_rank = greater_same_position + 1.0

    player_features = torch.stack(
        [
            projected_points / 500.0,
            projected_points / highest_available.unsqueeze(1),
            position_rank / candidate_same_position.clamp_min(1.0),
            candidate_same_position / 100.0,
            roster_size.unsqueeze(1).expand(-1, player_count) / 16.0,
            roster_points.unsqueeze(1).expand(-1, player_count) / 8000.0,
            *[
                roster_counts[:, position].unsqueeze(1).expand(-1, player_count) / 16.0
                for position in range(4)
            ],
            *[
                (positions == position).to(torch.float32).unsqueeze(0).expand(scenarios, -1)
                for position in range(4)
            ],
        ],
        dim=2,
    )

    starter_requirements = (1.0, 2.0, 2.0, 1.0)
    starter_needs = torch.stack(
        [
            ((requirement - roster_counts[:, position]).clamp_min(0.0) / requirement)
            for position, requirement in enumerate(starter_requirements)
        ],
        dim=1,
    )
    flex_count = (roster_positions != 0).sum(dim=1).to(torch.float32)
    roster_player_points = projected_points.gather(1, roster.clamp_min(0)).masked_fill(
        roster < 0, 0.0
    )
    starter_total = (
        roster_player_points.topk(k=min(7, roster_player_points.shape[1]), dim=1)
        .values.sum(dim=1)
    )
    bench_points = (roster_points - starter_total).clamp_min(0.0)
    state_values = torch.stack(
        [
            roster_size / 16.0,
            *[roster_counts[:, position] / 8.0 for position in range(4)],
            torch.zeros_like(roster_size),
            torch.zeros_like(roster_size),
            available_count / 250.0,
            *[
                available_position_counts[:, position] / available_count.clamp_min(1.0)
                for position in range(4)
            ],
            roster_points / 1000.0,
            highest_available / 500.0,
            torch.zeros_like(roster_size),
            torch.zeros_like(roster_size),
            torch.zeros_like(roster_size),
            torch.zeros_like(roster_size),
            torch.zeros_like(roster_size),
            *[starter_needs[:, position] for position in range(4)],
            flex_count / 16.0,
            bench_points / 500.0,
        ],
        dim=1,
    )
    state_features = state_values.unsqueeze(1).expand(-1, player_count, -1)
    return player_features, state_features


@torch.inference_mode()
def run_batched_policy_draft(
    projected_points: torch.Tensor,
    positions: torch.Tensor,
    policy_network: torch.nn.Module,
    *,
    policy_team_indices: torch.Tensor | None = None,
    team_count: int = 10,
    rounds: int = 16,
    anchor_weight: float = 0.20,
    use_amp: bool = True,
) -> DraftBatchResult:
    """Draft one policy-controlled team against projection-best opponents.

    ``policy_team_indices`` rotates the candidate's draft slot across scenarios,
    preventing a learned policy from overfitting to the first pick.
    """

    _validate_inputs(projected_points, positions, team_count, rounds)
    scenarios, player_count = projected_points.shape
    if policy_team_indices is None:
        policy_team_indices = torch.zeros(
            scenarios,
            dtype=torch.long,
            device=projected_points.device,
        )
    if policy_team_indices.shape != (scenarios,):
        raise ValueError("policy_team_indices must have shape [scenarios].")

    working_scores = projected_points.clone()
    available = torch.ones_like(projected_points, dtype=torch.bool)
    rosters = torch.full(
        (scenarios, team_count, rounds),
        -1,
        dtype=torch.long,
        device=projected_points.device,
    )
    position_one_hot = _position_one_hot(positions)
    minimums = projected_points.new_tensor((1, 4, 4, 1))
    maximums = projected_points.new_tensor((2, 6, 7, 3))
    autocast_enabled = use_amp and projected_points.device.type == "cuda"
    counts = torch.zeros(
        (scenarios, team_count, 4), dtype=torch.int16, device=projected_points.device
    )
    policy_network.eval()

    for round_number in range(rounds):
        order = range(team_count) if round_number % 2 == 0 else range(team_count - 1, -1, -1)
        for team_index in order:
            team_counts = counts[:, team_index]
            candidate_counts = team_counts.unsqueeze(1) + position_one_hot.unsqueeze(0)
            missing = (minimums - candidate_counts).clamp_min(0).sum(dim=2)
            remaining = rounds - round_number - 1
            shape_allowed = (candidate_counts <= maximums).all(dim=2) & (missing <= remaining)
            constrained = available & shape_allowed
            no_constrained = ~constrained.any(dim=1)

            player_features, state_features = _build_player_features(
                projected_points=working_scores.masked_fill(~available, 0.0),
                positions=positions,
                available=available,
                rosters=rosters,
                team_index=team_index,
            )
            flat_player = player_features.reshape(-1, player_features.shape[-1])
            flat_state = state_features.reshape(-1, state_features.shape[-1])
            with torch.autocast(
                device_type=projected_points.device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                policy_scores = policy_network(
                    flat_player,
                    flat_state,
                    decision_type="draft",
                ).reshape(scenarios, player_count)
            projection_anchor = working_scores / working_scores.masked_fill(
                ~available, float("-inf")
            ).max(dim=1).values.clamp_min(1.0).unsqueeze(1)
            scores = policy_scores.to(working_scores.dtype) + anchor_weight * projection_anchor
            scores = scores.masked_fill(~constrained, float("-inf"))
            opponent_scores = working_scores.masked_fill(~available, float("-inf"))
            use_policy = policy_team_indices == team_index
            scores = torch.where(use_policy.unsqueeze(1), scores, opponent_scores)
            scores = torch.where(no_constrained.unsqueeze(1), opponent_scores, scores)
            selected = scores.argmax(dim=1)
            rosters[:, team_index, round_number] = selected
            counts[:, team_index] += position_one_hot[selected].to(torch.int16)
            available.scatter_(1, selected.unsqueeze(1), False)
            working_scores.scatter_(1, selected.unsqueeze(1), float("-inf"))

    return DraftBatchResult(player_indices=rosters)
