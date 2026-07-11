from dataclasses import dataclass, field

from fantasy_engine.player import Player


@dataclass
class Team:
    name: str
    roster: list[Player] = field(default_factory=list)

    def add_player(self, player: Player) -> None:
        self.roster.append(player)

    def remove_player(self, player: Player) -> None:
        self.roster.remove(player)

    def has_player(self, player: Player) -> bool:
        return player in self.roster

    def roster_size(self) -> int:
        return len(self.roster)

    def projected_score(self) -> float:
        return sum(player.projected_score for player in self.roster)

    def actual_score(self) -> float:
        return sum(player.actual_score for player in self.roster)
