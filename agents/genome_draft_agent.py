from dataclasses import dataclass

from evolution.genome import DraftStrategyGenome
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team

DEFAULT_ROSTER_SIZE = 16
ROSTER_MINIMUMS = {
    "QB": 1,
    "RB": 4,
    "WR": 4,
    "TE": 1,
    "K": 1,
    "DST": 1,
}
ROSTER_MAXIMUMS = {
    "QB": 2,
    "RB": 6,
    "WR": 7,
    "TE": 3,
    "K": 1,
    "DST": 1,
}


@dataclass
class GenomeDraftAgent:
    genome: DraftStrategyGenome
    roster_size: int = DEFAULT_ROSTER_SIZE

    def choose_player(
        self,
        available_players: list[Player],
        team: Team,
        league: League,
    ) -> Player:
        if not available_players:
            raise ValueError("Cannot choose a player from an empty player pool.")

        eligible_players = self.get_eligible_players(
            available_players=available_players,
            team=team,
            league=league,
        )

        return max(
            eligible_players,
            key=lambda player: self.score_player(player, available_players),
        )

    def get_eligible_players(
        self,
        available_players: list[Player],
        team: Team,
        league: League,
    ) -> list[Player]:
        eligible_players = []
        minimum_context = None

        if self.should_enforce_minimum_roster_shape(league):
            minimum_context = self.get_league_minimum_context(league)

        for player in available_players:
            if self.can_draft_player(player, team, league, minimum_context):
                eligible_players.append(player)

        if not eligible_players:
            raise ValueError("No available players satisfy the roster construction rules.")

        return eligible_players

    def can_draft_player(
        self,
        player: Player,
        team: Team,
        league: League,
        minimum_context: tuple[dict[str, int], dict[str, int]] | None = None,
    ) -> bool:
        position_counts = self.get_position_counts(team)

        if (
            player.position in ROSTER_MAXIMUMS
            and position_counts[player.position] >= ROSTER_MAXIMUMS[player.position]
        ):
            return False

        if minimum_context is None:
            if not self.should_enforce_minimum_roster_shape(league):
                return True

            minimum_context = self.get_league_minimum_context(league)

        if player.position in position_counts:
            position_counts[player.position] += 1
        remaining_roster_spots = self.roster_size - team.roster_size() - 1
        missing_minimum_slots = self.get_missing_minimum_slots(position_counts)

        return (
            remaining_roster_spots >= missing_minimum_slots
            and self.has_enough_players_for_league_minimums(
                player,
                team,
                league,
                minimum_context,
            )
        )

    def get_position_counts(self, team: Team) -> dict[str, int]:
        position_counts = {position: 0 for position in ROSTER_MINIMUMS}

        for roster_player in team.roster:
            if roster_player.position in position_counts:
                position_counts[roster_player.position] += 1

        return position_counts

    def get_missing_minimum_slots(self, position_counts: dict[str, int]) -> int:
        missing_slots = 0

        for position, minimum_count in ROSTER_MINIMUMS.items():
            missing_slots += max(0, minimum_count - position_counts[position])

        return missing_slots

    def has_enough_players_for_league_minimums(
        self,
        selected_player: Player,
        selected_team: Team,
        league: League,
        minimum_context: tuple[dict[str, int], dict[str, int]],
    ) -> bool:
        missing_counts, available_counts = minimum_context
        updated_missing_counts = dict(missing_counts)
        updated_available_counts = dict(available_counts)

        selected_team_counts = self.get_position_counts(selected_team)

        if selected_player.position in selected_team_counts:
            if (
                selected_team_counts[selected_player.position]
                < ROSTER_MINIMUMS[selected_player.position]
            ):
                updated_missing_counts[selected_player.position] -= 1

            updated_available_counts[selected_player.position] -= 1

        for position, missing_count in updated_missing_counts.items():
            if updated_available_counts[position] < missing_count:
                return False

        return True

    def get_league_minimum_context(
        self,
        league: League,
    ) -> tuple[dict[str, int], dict[str, int]]:
        missing_counts = {position: 0 for position in ROSTER_MINIMUMS}
        available_counts = {position: 0 for position in ROSTER_MINIMUMS}

        for team in league.teams:
            position_counts = self.get_position_counts(team)

            for position, minimum_count in ROSTER_MINIMUMS.items():
                missing_counts[position] += max(0, minimum_count - position_counts[position])

        for player in league.available_players:
            if player.position in available_counts:
                available_counts[player.position] += 1

        return missing_counts, available_counts

    def should_enforce_minimum_roster_shape(self, league: League) -> bool:
        position_totals = {position: 0 for position in ROSTER_MINIMUMS}

        for team in league.teams:
            for player in team.roster:
                if player.position in position_totals:
                    position_totals[player.position] += 1

        for player in league.available_players:
            if player.position in position_totals:
                position_totals[player.position] += 1

        for position, minimum_count in ROSTER_MINIMUMS.items():
            if position_totals[position] < len(league.teams) * minimum_count:
                return False

        return True

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
        return player.projected_score

    def get_floor_score(self, player: Player) -> float:
        return player.projected_score

    def get_bye_penalty(self, player: Player) -> float:
        return 0.0
