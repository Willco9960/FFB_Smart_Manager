"""Train and evaluate separate manager checkpoints on chronological folds."""

import argparse
import json
from pathlib import Path

from agents.baseline_agents import create_baseline_opponents
from agents.neural_draft_agent import NeuralDraftAgent
from evolution.full_season import evaluate_full_season_battle_royale
from evolution.genome import DraftStrategyGenome, create_random_genome
from evolution.modular_policy_training import train_modular_policy_self_play
from evolution.walk_forward import build_walk_forward_folds
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES
from fantasy_engine.weekly_data import load_weekly_performances
from models.modular_manager_policy import (
    ModularManagerPolicyNetwork,
    load_modular_policy_network,
    save_modular_policy_network,
)
from scripts.train_modular_manager_policy import create_training_league


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2001)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--minimum-training-seasons", type=int, default=5)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--selection", type=int, default=3)
    parser.add_argument(
        "--initial-policy",
        type=Path,
        default=Path("data/models/modular_manager_policy.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models/modular_walk_forward"),
    )
    return parser.parse_args()


def load_transaction_genome() -> DraftStrategyGenome:
    path = Path("data/evolution/best_full_season_2021_genome.json")
    if path.exists():
        return DraftStrategyGenome.from_json(path.read_text(encoding="utf-8"))
    return create_random_genome(seed=2021)


def create_scenario(season: int):
    return (
        create_training_league(season),
        load_weekly_performances(season, include_special_teams=True),
    )


def evaluate_candidate(
    policy: ModularManagerPolicyNetwork,
    season: int,
    transaction_genome: DraftStrategyGenome,
    seed: int,
):
    candidate = NeuralDraftAgent(policy_network=policy, genome=transaction_genome)
    opponents = create_baseline_opponents(opponent_count=9, seed=seed)
    results = evaluate_full_season_battle_royale(
        agents=[candidate, *opponents],
        league=create_training_league(season),
        performances=load_weekly_performances(season, include_special_teams=True),
        lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
        seed=seed,
        transaction_genome_fallback=transaction_genome,
    )
    candidate_result = next(result for result in results if result.agent is candidate)
    return {
        "season": season,
        "fitness": candidate_result.fitness_score,
        "wins": candidate_result.regular_season_wins,
        "points_for": candidate_result.points_for,
        "playoff": candidate_result.playoff_rate > 0.0,
        "champion": candidate_result.champion,
    }


def main() -> None:
    args = parse_args()
    folds = build_walk_forward_folds(
        args.start_season,
        args.end_season,
        minimum_training_seasons=args.minimum_training_seasons,
    )
    if not args.initial_policy.exists():
        raise FileNotFoundError(
            f"Initial modular policy checkpoint not found: {args.initial_policy}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transaction_genome = load_transaction_genome()
    fold_results = []

    for fold_number, fold in enumerate(folds, start=1):
        print(
            f"Fold {fold_number}: train {fold.training_seasons} -> "
            f"validate {fold.validation_season} -> test {fold.test_season}",
            flush=True,
        )
        initial_policy = load_modular_policy_network(args.initial_policy)
        trained_policy, history = train_modular_policy_self_play(
            initial_policy=initial_policy,
            scenarios=[create_scenario(season) for season in fold.training_seasons],
            transaction_genome=transaction_genome,
            population_size=args.population,
            generations=args.generations,
            selection_count=args.selection,
            lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
            seed=fold_number,
        )
        checkpoint_path = save_modular_policy_network(
            trained_policy,
            args.output_dir / f"fold_{fold_number:03d}_test_{fold.test_season}.pt",
        )
        evaluation = evaluate_candidate(
            trained_policy,
            fold.test_season,
            transaction_genome,
            seed=1000 + fold_number,
        )
        evaluation["training_history"] = history
        evaluation["checkpoint"] = str(checkpoint_path)
        fold_results.append(evaluation)
        print(
            f"  test {fold.test_season}: fitness {evaluation['fitness']:.2f}, "
            f"wins {evaluation['wins']}, PF {evaluation['points_for']:.2f}, "
            f"playoff {evaluation['playoff']}",
            flush=True,
        )

    report_path = args.output_dir / "walk_forward_results.json"
    report_path.write_text(json.dumps(fold_results, indent=2), encoding="utf-8")
    print(f"Walk-forward report saved to: {report_path}")


if __name__ == "__main__":
    main()
