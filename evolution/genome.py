import json
import random
from dataclasses import asdict, dataclass

GENOME_WEIGHT_RANGES = {
    "projection_weight": (0.5, 1.0),
    "position_scarcity_weight": (0.0, 0.8),
    "adp_value_weight": (0.0, 1.0),
    "upside_weight": (0.0, 0.7),
    "floor_weight": (0.0, 0.7),
    "bye_week_penalty": (0.0, 0.3),
    "qb_priority": (0.0, 0.35),
    "rb_priority": (0.35, 1.0),
    "wr_priority": (0.35, 1.0),
    "te_priority": (0.1, 0.65),
}


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
        projection_weight=random_weight(rng, *GENOME_WEIGHT_RANGES["projection_weight"]),
        position_scarcity_weight=random_weight(
            rng,
            *GENOME_WEIGHT_RANGES["position_scarcity_weight"],
        ),
        adp_value_weight=random_weight(rng, *GENOME_WEIGHT_RANGES["adp_value_weight"]),
        upside_weight=random_weight(rng, *GENOME_WEIGHT_RANGES["upside_weight"]),
        floor_weight=random_weight(rng, *GENOME_WEIGHT_RANGES["floor_weight"]),
        bye_week_penalty=random_weight(rng, *GENOME_WEIGHT_RANGES["bye_week_penalty"]),
        qb_priority=random_weight(rng, *GENOME_WEIGHT_RANGES["qb_priority"]),
        rb_priority=random_weight(rng, *GENOME_WEIGHT_RANGES["rb_priority"]),
        wr_priority=random_weight(rng, *GENOME_WEIGHT_RANGES["wr_priority"]),
        te_priority=random_weight(rng, *GENOME_WEIGHT_RANGES["te_priority"]),
    )
