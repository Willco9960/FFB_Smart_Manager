from dataclasses import dataclass

from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from models.manager_policy_nn import ManagerPolicyNetwork, create_draft_action_features


@dataclass
class NeuralDraftAgent:
    policy_network: ManagerPolicyNetwork

    def choose_player(
        self,
        available_players: list[Player],
        team: Team,
        league: League,
    ) -> Player:
        if not available_players:
            raise ValueError("Cannot choose a player from an empty player pool.")

        return max(
            available_players,
            key=lambda player: self.policy_network.score_action(
                create_draft_action_features(player, team, available_players)
            ),
        )
