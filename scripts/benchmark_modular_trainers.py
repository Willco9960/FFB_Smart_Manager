"""Compare standard scenario-parallel training with island training."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from evolution.genome import DraftStrategyGenome, create_random_genome
from evolution.island_policy_training import train_island_policy_self_play
from evolution.modular_policy_training import train_modular_policy_self_play
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES
from models.modular_manager_policy import ModularManagerPolicyNetwork
from scripts.train_modular_policy_islands import create_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2022)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--selection", type=int, default=2)
    parser.add_argument("--evaluation-workers", type=int, default=4)
    parser.add_argument("--islands", type=int, default=2)
    parser.add_argument("--island-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/modular_trainer_benchmark.json"),
    )
    return parser.parse_args()


def load_transaction_genome() -> DraftStrategyGenome:
    path = Path("data/evolution/best_full_season_2021_genome.json")
    if path.exists():
        return DraftStrategyGenome.from_json(path.read_text(encoding="utf-8"))
    return create_random_genome(seed=2021)


def main() -> None:
    args = parse_args()
    scenarios = create_scenarios(args.start_season, args.end_season)
    transaction_genome = load_transaction_genome()
    results = {}

    torch.manual_seed(args.seed)
    standard_started = perf_counter()
    standard_policy, standard_history = train_modular_policy_self_play(
        initial_policy=ModularManagerPolicyNetwork(),
        scenarios=scenarios,
        transaction_genome=transaction_genome,
        population_size=args.population,
        generations=args.generations,
        selection_count=args.selection,
        seed=args.seed,
        lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
        scenarios_per_generation=None,
        evaluation_workers=args.evaluation_workers,
        run_final_evaluation=False,
    )
    results["standard"] = {
        "elapsed_seconds": round(perf_counter() - standard_started, 2),
        "best_score": max(standard_history),
        "final_score": standard_history[-1],
    }
    del standard_policy

    torch.manual_seed(args.seed)
    island_started = perf_counter()
    island_result = train_island_policy_self_play(
        initial_policy=ModularManagerPolicyNetwork(),
        scenarios=scenarios,
        transaction_genome=transaction_genome,
        island_count=args.islands,
        segments=1,
        generations_per_segment=args.generations,
        population_size=args.population,
        selection_count=args.selection,
        seed=args.seed,
        lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
        scenarios_per_generation=None,
        island_workers=args.island_workers,
    )
    results["island"] = {
        "elapsed_seconds": round(perf_counter() - island_started, 2),
        "best_score": island_result.best_score,
        "final_segment_average": round(
            sum(island_result.segment_scores[-1]) / len(island_result.segment_scores[-1]),
            2,
        ),
    }
    standard_time = results["standard"]["elapsed_seconds"]
    island_time = results["island"]["elapsed_seconds"]
    results["comparison"] = {
        "island_speedup_vs_standard": round(standard_time / island_time, 3)
        if island_time
        else None,
        "note": "Compare quality on the same chronological holdout before selecting a trainer.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Benchmark saved to: {args.output}")


if __name__ == "__main__":
    main()
