"""Deterministic transaction-action selection primitives shared by CPU/CUDA."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch

T = TypeVar("T")


def stable_best(  # noqa: UP047
    actions: Sequence[T],
    *,
    score: Callable[[T], float],
    tie_key: Callable[[T], object],
) -> T:
    """Select the highest score, breaking ties by the smallest explicit key."""
    if not actions:
        raise ValueError("stable_best requires at least one action")
    return min(actions, key=lambda action: (-float(score(action)), tie_key(action)))


def stable_argmax(scores: torch.Tensor, *, tie_break_indices: torch.Tensor) -> torch.Tensor:
    """Return argmax with explicit ascending tie-break indices."""
    if scores.ndim != 1 or tie_break_indices.shape != scores.shape:
        raise ValueError("scores and tie_break_indices must be matching vectors")
    order = sorted(
        range(scores.numel()),
        key=lambda index: (-float(scores[index].item()), int(tie_break_indices[index].item())),
    )
    return torch.tensor(order[0], dtype=torch.long, device=scores.device)


def stable_argmin(scores: torch.Tensor, *, tie_break_indices: torch.Tensor) -> torch.Tensor:
    """Return argmin with explicit ascending tie-break indices."""
    if scores.ndim != 1 or tie_break_indices.shape != scores.shape:
        raise ValueError("scores and tie_break_indices must be matching vectors")
    order = sorted(
        range(scores.numel()),
        key=lambda index: (float(scores[index].item()), int(tie_break_indices[index].item())),
    )
    return torch.tensor(order[0], dtype=torch.long, device=scores.device)


def stable_topk(
    scores: torch.Tensor,
    k: int,
    *,
    tie_break_indices: torch.Tensor,
) -> torch.Tensor:
    """Return top-k indices ordered by descending score then ascending key."""
    if scores.ndim != 1 or tie_break_indices.shape != scores.shape:
        raise ValueError("scores and tie_break_indices must be matching vectors")
    if k < 0 or k > scores.numel():
        raise ValueError("k must be within the score-vector size")
    order = sorted(
        range(scores.numel()),
        key=lambda index: (-float(scores[index].item()), int(tie_break_indices[index].item())),
    )[:k]
    return torch.tensor(order, dtype=torch.long, device=scores.device)



def canonical_player_key(player_id: str, position: str = "", team: str = "") -> str:
    """Return a backend-neutral stable key for a player identity."""
    return "|".join((str(player_id), str(position), str(team)))


def canonical_waiver_action_key(team_name: str, add_player_key: str, drop_player_key: str) -> str:
    return "waiver|" + "|".join((team_name, add_player_key, drop_player_key))


def canonical_trade_action_key(
    proposer: str,
    counterparty: str,
    offered_player_keys: Sequence[str],
    requested_player_keys: Sequence[str],
) -> str:
    offered = ",".join(sorted(offered_player_keys))
    requested = ",".join(sorted(requested_player_keys))
    return "trade|" + "|".join((proposer, counterparty, offered, requested))


def canonical_transaction_state_digest(
    *,
    team_rosters: Sequence[tuple[str, Sequence[str]]],
    available_player_keys: Sequence[str],
    locked_team_names: Sequence[str] = (),
    standings: Sequence[tuple[str, int, float]] = (),
) -> str:
    """Digest the complete sequential transaction state in backend-neutral form."""
    payload = {
        "team_rosters": tuple(
            (str(team_name), tuple(str(player_key) for player_key in roster))
            for team_name, roster in sorted(team_rosters, key=lambda item: str(item[0]))
        ),
        "available_player_keys": tuple(sorted(str(key) for key in available_player_keys)),
        "locked_team_names": tuple(sorted(str(name) for name in locked_team_names)),
        "standings": tuple(
            # League reports publish hundredths; finer precision would turn
            # harmless CUDA float32 accumulation noise into false divergence.
            (str(team_name), int(wins), round(float(points_for), 2))
            for team_name, wins, points_for in sorted(standings, key=lambda item: str(item[0]))
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TransactionEvent:
    """Canonical trace record for one sequential transaction decision."""

    season: int
    week: int
    sequence_index: int
    decision_type: str
    team_name: str
    action_key: str
    pre_state_digest: str
    post_state_digest: str
    accepted: bool = True
    rejection_reason: str = ""
    reward_components: tuple[tuple[str, float], ...] = ()
    player_keys: tuple[str, ...] = ()
    team_index: int | None = None
    counterparty_index: int | None = None

    def __post_init__(self) -> None:
        if self.decision_type not in ("waiver", "trade"):
            raise ValueError(f"Unknown transaction decision type: {self.decision_type}")
        if self.sequence_index < 0:
            raise ValueError("sequence_index must be non-negative")
        if not self.accepted and not self.rejection_reason:
            raise ValueError("Rejected transaction events require a rejection_reason")

    def digest(self) -> str:
        payload = {
            "season": self.season,
            "week": self.week,
            "sequence_index": self.sequence_index,
            "decision_type": self.decision_type,
            "team_name": self.team_name,
            "action_key": self.action_key,
            "pre_state_digest": self.pre_state_digest,
            "post_state_digest": self.post_state_digest,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "reward_components": self.reward_components,
            "player_keys": self.player_keys,
            "team_index": self.team_index,
            "counterparty_index": self.counterparty_index,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
