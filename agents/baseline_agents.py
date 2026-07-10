import random

from agents.genome_draft_agent import GenomeDraftAgent
from evolution.genome import DraftStrategyGenome
from fantasy_engine.draft import DraftAgent


def create_baseline_genomes() -> list[DraftStrategyGenome]:
    return [
        DraftStrategyGenome(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.5, 0.5, 0.2),
        DraftStrategyGenome(0.8, 0.3, 0.4, 0.2, 0.0, 0.0, 0.05, 1.0, 0.4, 0.2),
        DraftStrategyGenome(0.8, 0.3, 0.4, 0.2, 0.0, 0.0, 0.05, 0.4, 1.0, 0.2),
        DraftStrategyGenome(0.8, 0.7, 0.4, 0.2, 0.0, 0.0, 0.05, 0.5, 0.5, 0.65),
        DraftStrategyGenome(0.7, 0.5, 0.8, 0.3, 0.0, 0.0, 0.05, 0.6, 0.7, 0.3),
        DraftStrategyGenome(0.6, 0.4, 0.3, 0.7, 0.0, 0.0, 0.05, 0.7, 0.8, 0.3),
    ]


def create_baseline_opponents(
    opponent_count: int,
    seed: int,
) -> list[DraftAgent]:
    rng = random.Random(seed)
    genomes = create_baseline_genomes()
    opponents: list[DraftAgent] = []

    for _ in range(opponent_count):
        genome = genomes[rng.randrange(len(genomes))]
        opponents.append(GenomeDraftAgent(genome=genome))

    rng.shuffle(opponents)

    return opponents
