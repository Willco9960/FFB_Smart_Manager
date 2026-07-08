import random
from dataclasses import dataclass, field

from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team


@dataclass
class RandomDraftAgent:
    seed: int | None = None
    rng: random.Random = field(init=False)

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def choose_player(
        self,
        available_players: list[Player],
        team: Team,
        league: League,
    ) -> Player:
        if not available_players:
            raise ValueError("Cannot choose a player from an empty player pool.")

        return self.rng.choice(available_players)
