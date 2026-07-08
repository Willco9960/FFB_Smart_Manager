from dataclasses import dataclass, field

from fantasy_engine.player import Player
from fantasy_engine.team import Team


@dataclass
class League:
    name: str
    teams: list[Team] = field(default_factory=list)
    available_players: list[Player] = field(default_factory=list)
    roster_rules: dict[str, int] = field(default_factory=dict)

    def add_team(self, team: Team) -> None:
        self.teams.append(team)

    def add_available_player(self, player: Player) -> None:
        self.available_players.append(player)

    def team_count(self) -> int:
        return len(self.teams)

    def available_player_count(self) -> int:
        return len(self.available_players)
