import math
import random
from dataclasses import dataclass, field

from agents.decision_scoring import blend_policy_and_anchor_scores
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
    exploration_rate: float = 0.0
    exploration_top_k: int = 5
    random_seed: int | None = None
    _rng: random.Random = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be between zero and one.")
        if self.exploration_top_k < 1:
            raise ValueError("exploration_top_k must be at least one.")
        self._rng = random.Random(self.random_seed)

    def reset_episode(self, seed: int) -> None:
        """Reset per-season exploration so episode randomness is reproducible."""

        self._rng.seed(seed)

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
            policy_scores = self.policy_network.score_decisions(features, "draft")
            anchor_scores = [
                self._anchor_score(player, team, available_players) for player in eligible_players
            ]
            scores = blend_policy_and_anchor_scores(policy_scores, anchor_scores)
            ranked = sorted(
                zip(scores, eligible_players, strict=True),
                key=lambda item: item[0],
                reverse=True,
            )
            if self._rng.random() < self.exploration_rate and len(ranked) > 1:
                candidates = ranked[: min(self.exploration_top_k, len(ranked))]
                weights = [math.exp((score - candidates[0][0]) / 0.12) for score, _ in candidates]
                return self._rng.choices(
                    [player for _, player in candidates],
                    weights=weights,
                    k=1,
                )[0]
            return ranked[0][1]

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

    def _anchor_score(self, player, team, available_players) -> float:
        position_counts = {
            position: sum(roster_player.position == position for roster_player in team.roster)
            for position in ("QB", "RB", "WR", "TE")
        }
        starter_need = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
        need_bonus = max(
            0, starter_need.get(player.position, 0) - position_counts.get(player.position, 0)
        )
        if self.genome is None:
            return player.projected_score + (need_bonus * 5.0)

        priority = {
            "QB": self.genome.qb_priority,
            "RB": self.genome.rb_priority,
            "WR": self.genome.wr_priority,
            "TE": self.genome.te_priority,
        }.get(player.position, 0.0)
        return (
            player.projected_score * max(self.genome.projection_weight, 0.1)
            + priority
            + (need_bonus * 5.0)
        )
