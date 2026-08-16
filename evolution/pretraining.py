"""Teacher warm-start and pretraining gates for manager policies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from evolution.modular_behavior_cloning import (
    ModularImitationExample,
    train_modular_behavior_policy,
)
from evolution.offline_replay import DecisionReplayBuffer, train_offline_policy
from fantasy_engine.league import League
from fantasy_engine.team import Team
from models.modular_manager_policy import (
    DECISION_TYPES,
    ModularManagerPolicyNetwork,
    create_modular_policy_features,
)


@dataclass(frozen=True)
class PretrainingResult:
    example_count: int
    decision_type_counts: tuple[tuple[str, int], ...]
    behavior_loss: float
    offline_loss: float | None
    approved: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "example_count": self.example_count,
            "decision_type_counts": dict(self.decision_type_counts),
            "behavior_loss": self.behavior_loss,
            "offline_loss": self.offline_loss,
            "approved": self.approved,
        }


def _teacher_score(player, available_players: list, decision_type: str) -> float:
    if not available_players:
        return 0.0
    projections = [candidate.projected_score for candidate in available_players]
    high = max(max(projections), 1.0)
    same_position = [
        candidate
        for candidate in available_players
        if candidate.position == player.position
    ]
    position_high = max(
        (candidate.projected_score for candidate in same_position),
        default=1.0,
    )
    scarcity = (
        1.0 - (same_position.index(player) / max(len(same_position), 1))
        if player in same_position
        else 0.0
    )
    if decision_type == "lineup":
        return player.projected_score / high
    if decision_type == "waiver":
        return (
            0.70 * player.projected_score / high
            + 0.30 * player.projected_score / max(position_high, 1.0)
        )
    if decision_type == "trade":
        return 0.55 * player.projected_score / high + 0.45 * scarcity
    return 0.65 * player.projected_score / high + 0.35 * scarcity


def build_manager_teacher_examples(
    league: League,
    *,
    episodes: int = 1,
    rounds: int = 16,
) -> list[ModularImitationExample]:
    """Build leakage-safe examples for every action head.

    The teacher uses only pre-decision projections and roster scarcity.  It is
    deliberately transparent; self-play remains responsible for improving
    beyond this warm start.
    """

    if episodes < 1 or rounds < 1:
        raise ValueError("episodes and rounds must be positive.")
    examples: list[ModularImitationExample] = []
    for _ in range(episodes):
        team = Team(name="Teacher Team")
        available = list(league.available_players)
        for _round_number in range(rounds):
            if not available:
                break
            for decision_type in DECISION_TYPES:
                candidate_pool = (
                    available if decision_type == "draft" else (team.roster or available)
                )
                if not candidate_pool:
                    continue
                scores = [
                    _teacher_score(player, candidate_pool, decision_type)
                    for player in candidate_pool
                ]
                scale = max(max(scores), 1.0)
                examples.extend(
                    ModularImitationExample(
                        features=create_modular_policy_features(
                            player,
                            team,
                            available,
                            projection_floor=max(player.projected_score * 0.75, 0.0),
                            projection_median=player.projected_score,
                            projection_ceiling=player.projected_score * 1.25,
                            boom_probability=0.25,
                        ),
                        target_score=score / scale,
                        decision_type=decision_type,
                    )
                    for player, score in zip(candidate_pool, scores, strict=True)
                )
            selected = max(available, key=lambda player: _teacher_score(player, available, "draft"))
            team.add_player(selected)
            available.remove(selected)
    return examples


def run_manager_pretraining(
    model: ModularManagerPolicyNetwork,
    examples: list[ModularImitationExample],
    *,
    behavior_epochs: int = 25,
    replay_buffer: DecisionReplayBuffer | None = None,
    offline_epochs: int = 25,
    device: torch.device | str = "cpu",
) -> PretrainingResult:
    if not examples:
        raise ValueError("At least one pretraining example is required.")
    counts = {decision_type: 0 for decision_type in DECISION_TYPES}
    for example in examples:
        counts[example.decision_type] = counts.get(example.decision_type, 0) + 1
    missing_heads = [decision_type for decision_type, count in counts.items() if count == 0]
    if missing_heads:
        raise ValueError(f"Pretraining examples are missing heads: {missing_heads}")

    behavior_loss = train_modular_behavior_policy(
        model,
        examples,
        epochs=behavior_epochs,
        device=device,
    )
    offline_loss = None
    if replay_buffer is not None and replay_buffer.records:
        offline_loss = train_offline_policy(
            model,
            replay_buffer,
            epochs=offline_epochs,
            device=device,
        )
    approved = math.isfinite(behavior_loss) and (
        offline_loss is None or math.isfinite(offline_loss)
    )
    return PretrainingResult(
        example_count=len(examples),
        decision_type_counts=tuple(sorted(counts.items())),
        behavior_loss=behavior_loss,
        offline_loss=offline_loss,
        approved=approved,
    )
