import json
import random
from dataclasses import asdict, dataclass


@dataclass
class DraftStrategyGenome:
    projection_weight: float
    position_scarcity_weight: float
    adp_value_weight: float
    upside_weight: float
    floor_weight: float
    bye_week_penalty: float
    qb_priority: float
    rb_priority: float
    wr_priority: float
    te_priority: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, genome_data: dict[str, float]) -> "DraftStrategyGenome":
        return cls(**genome_data)

    @classmethod
    def from_json(cls, genome_json: str) -> "DraftStrategyGenome":
        genome_data = json.loads(genome_json)

        return cls.from_dict(genome_data)


def random_weight(
    rng: random.Random,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return round(rng.uniform(minimum, maximum), 4)


def create_random_genome(seed: int | None = None) -> DraftStrategyGenome:
    rng = random.Random(seed)

    return DraftStrategyGenome(
        projection_weight=random_weight(rng),
        position_scarcity_weight=random_weight(rng),
        adp_value_weight=random_weight(rng),
        upside_weight=random_weight(rng),
        floor_weight=random_weight(rng),
        bye_week_penalty=random_weight(rng),
        qb_priority=random_weight(rng),
        rb_priority=random_weight(rng),
        wr_priority=random_weight(rng),
        te_priority=random_weight(rng),
    )
