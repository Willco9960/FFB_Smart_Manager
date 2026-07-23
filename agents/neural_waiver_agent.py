from dataclasses import dataclass

from agents.decision_scoring import bounded_policy_score
from fantasy_engine.league import League
from fantasy_engine.lineup import (
    ESPN_OFFENSIVE_LINEUP_RULES,
    LineupSlot,
    build_best_starting_lineup,
)
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from fantasy_engine.transactions import WaiverClaim
from models.manager_policy_nn import ManagerPolicyNetwork, create_draft_action_features
from models.modular_manager_policy import create_modular_policy_features


@dataclass
class NeuralWaiverAgent:
    policy_network: ManagerPolicyNetwork
    # Avoid burning a waiver claim for a marginal projection change.  A small
    # amount of abstention is valuable over a 14-week season because each
    # transaction also changes future roster options.
    minimum_lineup_improvement: float = 1.5
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES

    def choose_waiver_claim(
        self,
        team: Team,
        available_players: list[Player],
        league: League,
        week: int,
    ) -> WaiverClaim | None:
        baseline_score = self.get_projected_lineup_score(team)

        if baseline_score is None:
            return None

        best_choice = None
        best_action_score = float("-inf")

        for add_player in available_players:
            for drop_player in team.roster:
                updated_roster = [player for player in team.roster if player != drop_player]
                updated_roster.append(add_player)
                updated_team = Team(name=team.name, roster=updated_roster)
                updated_score = self.get_projected_lineup_score(updated_team)

                if updated_score is None:
                    continue

                improvement = updated_score - baseline_score

                if improvement < self.minimum_lineup_improvement:
                    continue

                action_score = improvement + bounded_policy_score(
                    self._score_player(add_player, team, available_players, week)
                )

                if action_score > best_action_score:
                    best_action_score = action_score
                    best_choice = WaiverClaim(
                        team_name=team.name,
                        add_player=add_player,
                        drop_player=drop_player,
                        week=week,
                    )

        return best_choice

    def _score_player(
        self,
        player: Player,
        team: Team,
        available_players: list[Player],
        week: int,
    ) -> float:
        if hasattr(self.policy_network, "score_waiver_action"):
            return self.policy_network.score_waiver_action(
                create_modular_policy_features(player, team, available_players, week)
            )

        return self.policy_network.score_action(
            create_draft_action_features(player, team, available_players)
        )

    def get_projected_lineup_score(self, team: Team) -> float | None:
        lineup = build_best_starting_lineup(
            roster=team.roster,
            lineup_rules=self.lineup_rules,
            require_complete_lineup=False,
            selection_score_attribute="projected_score",
        )

        if not lineup.is_complete():
            return None

        return sum(player.projected_score for player in lineup.players)
