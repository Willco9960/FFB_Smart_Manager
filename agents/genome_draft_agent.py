from dataclasses import dataclass

from evolution.genome import DraftStrategyGenome
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")


@dataclass
class GenomeDraftAgent:
    genome: DraftStrategyGenome

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
            key=lambda player: self.score_player(player, available_players),
        )

    def score_player(
        self,
        player: Player,
        available_players: list[Player],
    ) -> float:
        projection_score = player.projected_score * self.genome.projection_weight
        scarcity_score = self.get_position_scarcity_score(player, available_players)
        value_score = self.get_position_value_score(player, available_players)
        priority_score = self.get_position_priority(player.position)
        upside_score = self.get_upside_score(player)
        floor_score = self.get_floor_score(player)
        bye_penalty = self.get_bye_penalty(player)

        return (
            projection_score
            + (scarcity_score * self.genome.position_scarcity_weight)
            + (value_score * self.genome.adp_value_weight)
            + priority_score
            + (upside_score * self.genome.upside_weight)
            + (floor_score * self.genome.floor_weight)
            - (bye_penalty * self.genome.bye_week_penalty)
        )

    def get_position_priority(self, position: str) -> float:
        if position == "QB":
            return self.genome.qb_priority

        if position == "RB":
            return self.genome.rb_priority

        if position == "WR":
            return self.genome.wr_priority

        if position == "TE":
            return self.genome.te_priority

        return 0.0

    def get_position_scarcity_score(
        self,
        player: Player,
        available_players: list[Player],
    ) -> float:
        same_position_count = 0

        for available_player in available_players:
            if available_player.position == player.position:
                same_position_count += 1

        if same_position_count == 0:
            return 0.0

        return 1.0 / same_position_count

    def get_position_value_score(
        self,
        player: Player,
        available_players: list[Player],
    ) -> float:
        same_position_players = []

        for available_player in available_players:
            if available_player.position == player.position:
                same_position_players.append(available_player)

        if not same_position_players:
            return 0.0

        average_projection = sum(
            available_player.projected_score for available_player in same_position_players
        ) / len(same_position_players)

        return player.projected_score - average_projection

    def get_upside_score(self, player: Player) -> float:
        return max(player.projected_score, player.actual_score)

    def get_floor_score(self, player: Player) -> float:
        if player.actual_score == 0.0:
            return player.projected_score

        return min(player.projected_score, player.actual_score)

    def get_bye_penalty(self, player: Player) -> float:
        return 0.0
