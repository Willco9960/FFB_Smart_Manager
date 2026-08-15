"""Evaluate modular manager policies on a chronological holdout season."""

import argparse
import json
from pathlib import Path

from evolution.modular_holdout import (
    evaluate_modular_policy_path,
    load_holdout_transaction_genome,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, default=None)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument(
        "--transaction-genome",
        type=Path,
        default=Path("data/evolution/best_full_season_2021_genome.json"),
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transaction_genome = load_holdout_transaction_genome(args.transaction_genome)
    candidate = evaluate_modular_policy_path(
        model_path=args.model,
        label="candidate",
        season=args.holdout_season,
        transaction_genome=transaction_genome,
        seed=args.seed,
    )
    results = [candidate.to_dict()]
    print("Modular policy chronological holdout evaluation")
    print(f"Holdout season: {args.holdout_season}")
    print(
        f"Candidate: fitness={candidate.fitness:.2f} wins={candidate.wins:.2f} "
        f"PF={candidate.points_for:.2f} playoffs={candidate.playoff_rate:.1%} "
        f"championships={candidate.championship_rate:.1%}"
    )

    if args.baseline_model is not None:
        baseline = evaluate_modular_policy_path(
            model_path=args.baseline_model,
            label="baseline",
            season=args.holdout_season,
            transaction_genome=transaction_genome,
            seed=args.seed,
        )
        results.insert(0, baseline.to_dict())
        print(
            f"Baseline:  fitness={baseline.fitness:.2f} wins={baseline.wins:.2f} "
            f"PF={baseline.points_for:.2f} playoffs={baseline.playoff_rate:.1%} "
            f"championships={baseline.championship_rate:.1%}"
        )
        print(
            f"Delta: fitness={candidate.fitness - baseline.fitness:+.2f} "
            f"wins={candidate.wins - baseline.wins:+.2f} "
            f"PF={candidate.points_for - baseline.points_for:+.2f}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
