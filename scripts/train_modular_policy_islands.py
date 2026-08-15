"""Train modular manager policies with parallel evolutionary islands."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from evolution.genome import DraftStrategyGenome, create_random_genome
from evolution.island_policy_training import train_island_policy_self_play
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import load_weekly_performances
from models.modular_manager_policy import (
    ModularManagerPolicyNetwork,
    load_modular_policy_network,
    save_modular_policy_network,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2001)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--islands", type=int, default=10)
    parser.add_argument("--island-workers", type=int, default=10)
    parser.add_argument("--segments", type=int, default=10)
    parser.add_argument("--generations-per-segment", type=int, default=10)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--selection", type=int, default=8)
    parser.add_argument("--scenarios-per-generation", type=int, default=8)
    parser.add_argument("--full-evaluation-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--initial-policy",
        type=Path,
        default=Path("data/models/modular_manager_policy.pt"),
        help="Pretrained modular policy used to initialize every island.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/modular_manager_policy_islands.pt"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/modular_island_training_report.json"),
    )
    return parser.parse_args()


def create_scenarios(start_season: int, end_season: int):
    scenarios = []
    for season in range(start_season, end_season + 1):
        players = load_leakage_safe_player_pool(
            projection_season=season - 1,
            actual_season=season,
            include_special_teams=True,
        )[:250]
        league = League(
            name=f"Island Training League {season}",
            teams=[Team(name=f"Island Team {number}") for number in range(1, 11)],
            available_players=players,
        )
        scenarios.append((league, load_weekly_performances(season, include_special_teams=True)))
    return scenarios


def main() -> None:
    args = parse_args()
    if args.selection > args.population:
        raise ValueError("selection cannot exceed population.")
    started = perf_counter()
    scenarios = create_scenarios(args.start_season, args.end_season)
    transaction_genome_path = Path("data/evolution/best_full_season_2021_genome.json")
    transaction_genome = (
        DraftStrategyGenome.from_json(transaction_genome_path.read_text(encoding="utf-8"))
        if transaction_genome_path.exists()
        else create_random_genome(seed=2021)
    )
    initial_policy = (
        load_modular_policy_network(args.initial_policy)
        if args.initial_policy.exists()
        else ModularManagerPolicyNetwork()
    )
    print(f"Initial policy: {args.initial_policy if args.initial_policy.exists() else 'random'}")
    result = train_island_policy_self_play(
        initial_policy=initial_policy,
        scenarios=scenarios,
        transaction_genome=transaction_genome,
        island_count=args.islands,
        segments=args.segments,
        generations_per_segment=args.generations_per_segment,
        population_size=args.population,
        selection_count=args.selection,
        seed=args.seed,
        lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
        scenarios_per_generation=args.scenarios_per_generation,
        full_evaluation_interval=args.full_evaluation_every,
        island_workers=args.island_workers,
    )
    model_path = save_modular_policy_network(result.best_policy, args.output)
    report = {
        "status": "completed",
        "start_season": args.start_season,
        "end_season": args.end_season,
        "islands": args.islands,
        "island_workers": args.island_workers,
        "segments": args.segments,
        "generations_per_segment": args.generations_per_segment,
        "population": args.population,
        "selection": args.selection,
        "scenarios_per_generation": args.scenarios_per_generation,
        "full_evaluation_every": args.full_evaluation_every,
        "best_score": result.best_score,
        "segment_scores": result.segment_scores,
        "island_scores": result.island_scores,
        "model_path": str(model_path),
        "elapsed_seconds": round(perf_counter() - started, 2),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for segment_number, scores in enumerate(result.segment_scores, start=1):
        print(
            f"Segment {segment_number}/{args.segments}: "
            f"best={max(scores):.2f} average={sum(scores) / len(scores):.2f}",
            flush=True,
        )
    print("Island training complete", flush=True)
    print(f"Best score: {result.best_score:.2f}", flush=True)
    print(f"Policy saved to: {model_path}", flush=True)
    print(f"Report saved to: {args.report}", flush=True)
    print(f"Elapsed seconds: {report['elapsed_seconds']}", flush=True)


if __name__ == "__main__":
    main()
