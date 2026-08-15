"""Chronological holdout evaluation for modular manager policies."""

from dataclasses import asdict, dataclass
from pathlib import Path

from agents.baseline_agents import create_baseline_opponents
from agents.neural_draft_agent import NeuralDraftAgent
from evolution.full_season import evaluate_full_season_battle_royale
from evolution.genome import DraftStrategyGenome, create_random_genome
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import load_weekly_performances
from models.modular_manager_policy import (
    ModularManagerPolicyNetwork,
    load_modular_policy_network,
)


@dataclass(frozen=True)
class ModularHoldoutResult:
    label: str
    season: int
    fitness: float
    wins: float
    points_for: float
    playoff_rate: float
    championship_rate: float
    transaction_reward: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def create_holdout_league(season: int) -> League:
    players = load_leakage_safe_player_pool(
        projection_season=season - 1,
        actual_season=season,
        include_special_teams=True,
    )[:250]
    return League(
        name=f"{season} Modular Holdout League",
        teams=[Team(name=f"Holdout Team {number}") for number in range(1, 11)],
        available_players=players,
    )


def load_holdout_transaction_genome(path: Path) -> DraftStrategyGenome:
    if path.exists():
        return DraftStrategyGenome.from_json(path.read_text(encoding="utf-8"))
    return create_random_genome(seed=2021)


def evaluate_modular_policy(
    policy: ModularManagerPolicyNetwork,
    label: str,
    season: int,
    transaction_genome: DraftStrategyGenome,
    seed: int = 1,
    rounds: int = 16,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
) -> ModularHoldoutResult:
    candidate = NeuralDraftAgent(policy_network=policy, genome=transaction_genome)
    opponents = create_baseline_opponents(opponent_count=9, seed=seed + 50_000)
    league = create_holdout_league(season)
    performances = load_weekly_performances(season, include_special_teams=True)
    results = evaluate_full_season_battle_royale(
        agents=[candidate, *opponents],
        league=league,
        performances=performances,
        rounds=rounds,
        lineup_rules=lineup_rules,
        seed=seed,
        transaction_genome_fallback=transaction_genome,
        transaction_mode="genome",
    )
    result = next(item for item in results if item.agent is candidate)
    return ModularHoldoutResult(
        label=label,
        season=season,
        fitness=round(result.fitness_score, 2),
        wins=round(result.regular_season_wins, 2),
        points_for=round(result.points_for, 2),
        playoff_rate=round(result.playoff_rate, 4),
        championship_rate=round(result.championship_rate, 4),
        transaction_reward=round(result.transaction_reward, 2),
    )


def evaluate_modular_policy_path(
    model_path: Path,
    label: str,
    season: int,
    transaction_genome: DraftStrategyGenome,
    seed: int = 1,
) -> ModularHoldoutResult:
    return evaluate_modular_policy(
        policy=load_modular_policy_network(model_path),
        label=label,
        season=season,
        transaction_genome=transaction_genome,
        seed=seed,
    )
