from dataclasses import dataclass
from typing import Protocol

from fantasy_engine.lineup import (
    ESPN_OFFENSIVE_LINEUP_RULES,
    LineupSlot,
    StartingLineup,
    build_best_starting_lineup,
)
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from models.manager_policy_nn import ManagerPolicyNetwork, create_draft_action_features


class LineupAgent(Protocol):
    def choose_lineup(
        self,
        roster: list[Player],
    ) -> StartingLineup: ...


@dataclass
class NeuralLineupAgent:
    policy_network: ManagerPolicyNetwork
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES

    def choose_lineup(self, roster: list[Player]) -> StartingLineup:
        team = Team(name="Lineup Decision", roster=roster)
        selection_scores = {
            id(player): self.policy_network.score_action(
                create_draft_action_features(player, team, roster)
            )
            for player in roster
        }

        return build_best_starting_lineup(
            roster=roster,
            lineup_rules=self.lineup_rules,
            selection_scores=selection_scores,
        )
