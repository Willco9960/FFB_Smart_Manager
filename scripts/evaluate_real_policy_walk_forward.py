import argparse
import statistics
from pathlib import Path

from scripts.evaluate_synthetic_policy_real_seasons import evaluate_policy, load_transaction_genome

ORIGINAL_POLICY_PATH = Path("data/models/manager_policy_full_season.pt")
TRAINED_POLICY_PATH = Path("data/models/manager_policy_real_seasons.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate original and trained manager policies across multiple seasons."
    )
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--trained-policy", type=Path, default=TRAINED_POLICY_PATH)
    return parser.parse_args()


def format_policy_summary(label: str, results) -> str:
    fitness_values = [result.fitness for result in results]
    wins = [result.wins for result in results]
    points_for = [result.points_for for result in results]
    playoff_count = sum(result.playoff_seed is not None for result in results)
    championship_count = sum(result.champion for result in results)

    return (
        f"{label}: average fitness {statistics.mean(fitness_values):.2f}, "
        f"fitness stddev {statistics.pstdev(fitness_values):.2f}, "
        f"average wins {statistics.mean(wins):.2f}, "
        f"average PF {statistics.mean(points_for):.2f}, "
        f"playoffs {playoff_count}/{len(results)}, "
        f"championships {championship_count}/{len(results)}"
    )


def main() -> None:
    args = parse_args()

    if not ORIGINAL_POLICY_PATH.exists():
        raise FileNotFoundError(f"Original policy not found: {ORIGINAL_POLICY_PATH}")
    if not args.trained_policy.exists():
        raise FileNotFoundError(f"Trained policy not found: {args.trained_policy}")
    if args.end_season < args.start_season:
        raise ValueError("End season must be greater than or equal to start season.")

    transaction_genome = load_transaction_genome()
    seasons = range(args.start_season, args.end_season + 1)
    original_results = []
    trained_results = []

    for season in seasons:
        print(f"Evaluating season {season}...", flush=True)
        original = evaluate_policy(ORIGINAL_POLICY_PATH, season, transaction_genome)
        trained = evaluate_policy(args.trained_policy, season, transaction_genome)
        original_results.append(original)
        trained_results.append(trained)
        print(
            f"{season}: wins {original.wins}->{trained.wins}, "
            f"PF {original.points_for:.2f}->{trained.points_for:.2f}, "
            f"playoff {original.playoff_seed is not None}->{trained.playoff_seed is not None}, "
            f"champion {original.champion}->{trained.champion}",
            flush=True,
        )

    print("\nMulti-season policy evaluation")
    print(format_policy_summary("Original", original_results))
    print(format_policy_summary("Trained", trained_results))

    fitness_delta = statistics.mean(
        trained.fitness - original.fitness
        for original, trained in zip(original_results, trained_results, strict=True)
    )
    wins_delta = statistics.mean(
        trained.wins - original.wins
        for original, trained in zip(original_results, trained_results, strict=True)
    )
    points_delta = statistics.mean(
        trained.points_for - original.points_for
        for original, trained in zip(original_results, trained_results, strict=True)
    )
    print(
        f"Average trained-minus-original delta: fitness {fitness_delta:+.2f}, "
        f"wins {wins_delta:+.2f}, PF {points_delta:+.2f}"
    )
    print(
        "Note: this evaluates one trained checkpoint across multiple seasons. "
        "Strict walk-forward retraining would require a separate checkpoint per cutoff season."
    )


if __name__ == "__main__":
    main()
