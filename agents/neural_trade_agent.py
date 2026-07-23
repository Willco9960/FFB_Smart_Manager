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
from fantasy_engine.transactions import TradeProposal
from models.manager_policy_nn import ManagerPolicyNetwork, create_draft_action_features
from models.modular_manager_policy import create_modular_policy_features


@dataclass
class NeuralTradeAgent:
    policy_network: ManagerPolicyNetwork
    # Trades carry opportunity cost and can help the opponent, so require a
    # meaningful projected improvement on both sides before proposing one.
    minimum_lineup_improvement: float = 2.0
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES

    def choose_trade_proposal(
        self,
        team: Team,
        opposing_teams: list[Team],
        league: League,
        week: int,
    ) -> TradeProposal | None:
        baseline_score = self.get_projected_lineup_score(team)

        if baseline_score is None:
            return None

        best_proposal = None
        best_action_score = float("-inf")

        for opposing_team in opposing_teams:
            opposing_baseline_score = self.get_projected_lineup_score(opposing_team)

            if opposing_baseline_score is None:
                continue

            for offered_player in team.roster:
                for requested_player in opposing_team.roster:
                    updated_team = Team(
                        name=team.name,
                        roster=[player for player in team.roster if player != offered_player]
                        + [requested_player],
                    )
                    updated_opposing_team = Team(
                        name=opposing_team.name,
                        roster=[
                            player for player in opposing_team.roster if player != requested_player
                        ]
                        + [offered_player],
                    )
                    updated_team_score = self.get_projected_lineup_score(updated_team)
                    updated_opposing_score = self.get_projected_lineup_score(updated_opposing_team)

                    if updated_team_score is None or updated_opposing_score is None:
                        continue

                    team_improvement = updated_team_score - baseline_score
                    opposing_improvement = updated_opposing_score - opposing_baseline_score

                    if (
                        team_improvement < self.minimum_lineup_improvement
                        or opposing_improvement < self.minimum_lineup_improvement
                    ):
                        continue

                    action_score = (
                        team_improvement
                        + opposing_improvement
                        + bounded_policy_score(
                            self._score_player(requested_player, team, team.roster, week)
                        )
                        + bounded_policy_score(
                            self._score_player(
                                offered_player,
                                opposing_team,
                                opposing_team.roster,
                                week,
                            )
                        )
                    )

                    if action_score > best_action_score:
                        best_action_score = action_score
                        best_proposal = TradeProposal(
                            proposing_team_name=team.name,
                            receiving_team_name=opposing_team.name,
                            offered_players=(offered_player,),
                            requested_players=(requested_player,),
                            week=week,
                        )

        return best_proposal

    def _score_player(
        self,
        player: Player,
        team: Team,
        roster: list[Player],
        week: int,
    ) -> float:
        if hasattr(self.policy_network, "score_trade_action"):
            return self.policy_network.score_trade_action(
                create_modular_policy_features(player, team, roster, week)
            )

        return self.policy_network.score_action(create_draft_action_features(player, team, roster))

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
