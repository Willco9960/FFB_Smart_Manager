from dataclasses import dataclass

from agents.genome_draft_agent import GenomeDraftAgent
from evolution.genome import DraftStrategyGenome
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from models.manager_policy_nn import ManagerPolicyNetwork, create_draft_action_features
from models.modular_manager_policy import create_modular_policy_features


@dataclass
class NeuralDraftAgent:
    policy_network: ManagerPolicyNetwork
    genome: DraftStrategyGenome | None = None

    def choose_player(
        self,
        available_players: list[Player],
        team: Team,
        league: League,
    ) -> Player:
        if not available_players:
            raise ValueError("Cannot choose a player from an empty player pool.")

        eligible_players = available_players

        if self.genome is not None:
            eligible_players = GenomeDraftAgent(genome=self.genome).get_eligible_players(
                available_players=available_players,
                team=team,
                league=league,
            )

        if hasattr(self.policy_network, "score_decisions"):
            features = [
                create_modular_policy_features(player, team, available_players)
                for player in eligible_players
            ]
            scores = self.policy_network.score_decisions(features, "draft")
            return max(zip(scores, eligible_players, strict=True), key=lambda item: item[0])[1]

        return max(
            eligible_players,
            key=lambda player: self._score_player(player, team, available_players),
        )

    def _score_player(
        self,
        player: Player,
        team: Team,
        available_players: list[Player],
    ) -> float:
        if hasattr(self.policy_network, "score_draft_action"):
            return self.policy_network.score_draft_action(
                create_modular_policy_features(player, team, available_players)
            )

        return self.policy_network.score_action(
            create_draft_action_features(player, team, available_players)
        )
