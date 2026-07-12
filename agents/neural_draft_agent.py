from dataclasses import dataclass

from agents.genome_draft_agent import GenomeDraftAgent
from evolution.genome import DraftStrategyGenome
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from models.manager_policy_nn import ManagerPolicyNetwork, create_draft_action_features


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

        return max(
            eligible_players,
            key=lambda player: self.policy_network.score_action(
                create_draft_action_features(player, team, available_players)
            ),
        )
