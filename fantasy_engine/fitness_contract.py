"""Shared, versioned reward and league semantics for CPU and CUDA training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from fantasy_engine.fantasy_points import STANDARD_SCORING, FantasyScoringSettings
from fantasy_engine.lineup import ESPN_DEFAULT_LINEUP_RULES, LineupSlot
from fantasy_engine.season import ESPN_TEN_TEAM_DEFAULT_RULES, ESPNLeagueRules


@dataclass(frozen=True)
class FitnessContract:
    contract_version: str = "espn-fitness-v1"
    league_rules: ESPNLeagueRules = ESPN_TEN_TEAM_DEFAULT_RULES
    lineup_rules: tuple[LineupSlot, ...] = ESPN_DEFAULT_LINEUP_RULES
    scoring_settings: FantasyScoringSettings = field(
        default_factory=lambda: FantasyScoringSettings(**asdict(STANDARD_SCORING))
    )
    weekly_win_reward: float = 15.0
    points_for_weight: float = 0.05
    playoff_qualification_reward: float = 40.0
    playoff_win_reward: float = 30.0
    championship_reward: float = 150.0
    transaction_reward_weight: float = 0.25
    lineup_efficiency_weight: float = 0.05
    replacement_value_weight: float = 0.05
    invalid_action_penalty: float = 25.0
    variance_penalty: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "league_rules": asdict(self.league_rules),
            "lineup_rules": [asdict(slot) for slot in self.lineup_rules],
            "scoring_settings": asdict(self.scoring_settings),
            "weekly_win_reward": self.weekly_win_reward,
            "points_for_weight": self.points_for_weight,
            "playoff_qualification_reward": self.playoff_qualification_reward,
            "playoff_win_reward": self.playoff_win_reward,
            "championship_reward": self.championship_reward,
            "transaction_reward_weight": self.transaction_reward_weight,
            "lineup_efficiency_weight": self.lineup_efficiency_weight,
            "replacement_value_weight": self.replacement_value_weight,
            "invalid_action_penalty": self.invalid_action_penalty,
            "variance_penalty": self.variance_penalty,
        }

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


ESPN_FITNESS_CONTRACT = FitnessContract()
