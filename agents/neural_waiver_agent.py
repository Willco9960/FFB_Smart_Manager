from dataclasses import dataclass

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


@dataclass
class NeuralWaiverAgent:
    policy_network: ManagerPolicyNetwork
    minimum_lineup_improvement: float = 0.5
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
                updated_roster = [
                    player for player in team.roster if player != drop_player
                ]
                updated_roster.append(add_player)
                updated_team = Team(name=team.name, roster=updated_roster)
                updated_score = self.get_projected_lineup_score(updated_team)

                if updated_score is None:
                    continue

                improvement = updated_score - baseline_score

                if improvement < self.minimum_lineup_improvement:
                    continue

                action_score = self.policy_network.score_action(
                    create_draft_action_features(add_player, team, available_players)
                ) + improvement

                if action_score > best_action_score:
                    best_action_score = action_score
                    best_choice = WaiverClaim(
                        team_name=team.name,
                        add_player=add_player,
                        drop_player=drop_player,
                        week=week,
                    )

        return best_choice

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
