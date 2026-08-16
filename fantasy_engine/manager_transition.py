"""Shared state/action/transition records for CPU and CUDA policies.

The simulator remains the authority for legality and scoring.  This module is
the small, serializable protocol that both backends use to describe a policy
decision, its legal action mask, and the resulting transition.  Keeping the
protocol independent from either backend prevents the training loop from
silently learning a different state representation on CPU and CUDA.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from fantasy_engine.fitness_contract import ESPN_FITNESS_CONTRACT
from fantasy_engine.player import Player
from fantasy_engine.team import Team

DECISION_TYPES = ("draft", "lineup", "waiver", "trade")


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def player_key(player: Player) -> str:
    return player.player_id or f"{player.name}|{player.position}|{player.team}"


@dataclass(frozen=True)
class ManagerState:
    season: int
    week: int
    team_name: str
    roster_player_ids: tuple[str, ...]
    available_player_ids: tuple[str, ...]
    opponent_roster_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()
    standing_wins: int = 0
    standing_points_for: float = 0.0
    projected_points: float = 0.0
    projection_floor: float = 0.0
    projection_median: float = 0.0
    projection_ceiling: float = 0.0
    boom_probability: float = 0.0
    contract_digest: str = ESPN_FITNESS_CONTRACT.digest()

    def digest(self) -> str:
        return _digest(self.__dict__)


@dataclass(frozen=True)
class LegalActionMask:
    decision_type: str
    allowed_action_keys: tuple[str, ...]
    contract_digest: str = ESPN_FITNESS_CONTRACT.digest()

    def __post_init__(self) -> None:
        if self.decision_type not in DECISION_TYPES:
            raise ValueError(f"Unknown decision type: {self.decision_type}")

    def allows(self, action_key: str) -> bool:
        return action_key in self.allowed_action_keys

    def digest(self) -> str:
        return _digest(self.__dict__)


@dataclass(frozen=True)
class ManagerAction:
    decision_type: str
    action_key: str
    team_name: str
    player_ids: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.decision_type not in DECISION_TYPES:
            raise ValueError(f"Unknown decision type: {self.decision_type}")


@dataclass(frozen=True)
class ManagerTransition:
    state: ManagerState
    action: ManagerAction
    action_mask: LegalActionMask
    next_state: ManagerState
    reward_components: tuple[tuple[str, float], ...] = ()

    def validate(self) -> None:
        if self.action.decision_type != self.action_mask.decision_type:
            raise ValueError("Action and legal mask decision types do not match.")
        if not self.action_mask.allows(self.action.action_key):
            raise ValueError("Action is not allowed by the supplied legal mask.")
        if self.state.contract_digest != self.next_state.contract_digest:
            raise ValueError("State transition changed the fitness contract.")
        if self.state.contract_digest != self.action_mask.contract_digest:
            raise ValueError("State and action mask use different contracts.")

    def digest(self) -> str:
        self.validate()
        return _digest(
            {
                "state": self.state.digest(),
                "action": self.action.__dict__,
                "action_mask": self.action_mask.digest(),
                "next_state": self.next_state.digest(),
                "reward_components": self.reward_components,
            }
        )


def build_manager_state(
    team: Team,
    available_players: list[Player],
    *,
    season: int = 0,
    week: int = 0,
    opponent_rosters: Mapping[str, list[Player]] | None = None,
    standing_wins: int = 0,
    standing_points_for: float = 0.0,
    projection_floor: float = 0.0,
    projection_median: float | None = None,
    projection_ceiling: float = 0.0,
    boom_probability: float = 0.0,
    contract_digest: str = ESPN_FITNESS_CONTRACT.digest(),
) -> ManagerState:
    median = team.projected_score() if projection_median is None else projection_median
    opponents = tuple(
        sorted(
            (name, tuple(player_key(player) for player in roster))
            for name, roster in (opponent_rosters or {}).items()
        )
    )
    return ManagerState(
        season=season,
        week=week,
        team_name=team.name,
        roster_player_ids=tuple(player_key(player) for player in team.roster),
        available_player_ids=tuple(player_key(player) for player in available_players),
        opponent_roster_ids=opponents,
        standing_wins=standing_wins,
        standing_points_for=round(standing_points_for, 4),
        projected_points=round(team.projected_score(), 4),
        projection_floor=round(projection_floor, 4),
        projection_median=round(median, 4),
        projection_ceiling=round(projection_ceiling, 4),
        boom_probability=round(boom_probability, 6),
        contract_digest=contract_digest,
    )
