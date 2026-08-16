from dataclasses import dataclass


@dataclass
class Player:
    name: str
    position: str
    team: str
    projected_score: float = 0.0
    actual_score: float = 0.0
    player_id: str | None = None
    history_missing: bool = False

    def projection_error(self) -> float:
        return self.actual_score - self.projected_score
