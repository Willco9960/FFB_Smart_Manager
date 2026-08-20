"""Evaluate a frozen CUDA policy across independent deterministic holdout seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from gpu_sim.historical_adapter import create_historical_cuda_inputs
from gpu_sim.policy_training import evaluate_cuda_policy
from models.modular_manager_policy import load_modular_policy_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--initial-policy", type=Path, required=True)
    parser.add_argument("--holdout-seasons", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 2001, 3001, 4001, 5001])
    parser.add_argument("--players", type=int, default=256)
    parser.add_argument("--scenario-repeats", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--transactions", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cuda_multi_seed_evaluation.json"),
    )
    return parser.parse_args()


def evaluate(policy, state, *, seed: int, repeats: int, transactions: bool):
    result = evaluate_cuda_policy(
        policy,
        [state],
        scenario_repeats=repeats,
        projection_noise=0.015,
        enable_transactions=transactions,
        seed=seed,
        compile_policy=False,
    )
    return {
        "fitness": result.fitness,
        "fitness_stddev": result.fitness_stddev,
        "risk_adjusted_fitness": result.risk_adjusted_fitness,
        "wins": result.wins,
        "points_for": result.points_for,
        "playoff_rate": result.playoff_rate,
        "championship_rate": result.championship_rate,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if not args.seeds:
        raise SystemExit("At least one seed is required.")
    if len(set(args.holdout_seasons)) != len(args.holdout_seasons):
        raise SystemExit("Holdout seasons must be unique.")

    device = torch.device(args.device)
    candidate = load_modular_policy_network(args.policy).to(device)
    initial = load_modular_policy_network(args.initial_policy).to(device)
    rows = []
    for season in args.holdout_seasons:
        state = create_historical_cuda_inputs(
            season=season,
            players=args.players,
            device=device,
        ).state
        for seed in args.seeds:
            candidate_result = evaluate(
                candidate,
                state,
                seed=seed,
                repeats=args.scenario_repeats,
                transactions=args.transactions,
            )
            initial_result = evaluate(
                initial,
                state,
                seed=seed,
                repeats=args.scenario_repeats,
                transactions=args.transactions,
            )
            rows.append(
                {
                    "season": season,
                    "seed": seed,
                    "candidate": candidate_result,
                    "initial_policy": initial_result,
                    "delta_vs_initial": candidate_result["fitness"] - initial_result["fitness"],
                    "risk_adjusted_delta_vs_initial": (
                        candidate_result["risk_adjusted_fitness"]
                        - initial_result["risk_adjusted_fitness"]
                    ),
                }
            )

    by_season = {}
    for season in args.holdout_seasons:
        season_rows = [row for row in rows if row["season"] == season]
        deltas = [row["delta_vs_initial"] for row in season_rows]
        risk_deltas = [row["risk_adjusted_delta_vs_initial"] for row in season_rows]
        by_season[str(season)] = {
            "seed_count": len(season_rows),
            "mean_delta_vs_initial": mean(deltas),
            "min_delta_vs_initial": min(deltas),
            "mean_risk_adjusted_delta_vs_initial": mean(risk_deltas),
            "min_risk_adjusted_delta_vs_initial": min(risk_deltas),
            "candidate_beats_initial_every_seed": all(delta > 0 for delta in deltas),
            "candidate_risk_adjusted_beats_initial_every_seed": all(
                delta > 0 for delta in risk_deltas
            ),
        }

    report = {
        "policy": str(args.policy),
        "initial_policy": str(args.initial_policy),
        "policy_sha256": sha256_file(args.policy),
        "initial_policy_sha256": sha256_file(args.initial_policy),
        "holdout_seasons": args.holdout_seasons,
        "seeds": args.seeds,
        "players": args.players,
        "scenario_repeats": args.scenario_repeats,
        "transactions": args.transactions,
        "rows": rows,
        "by_season": by_season,
        "promotion_ready_multi_seed": bool(by_season)
        and all(
            item["candidate_beats_initial_every_seed"]
            and item["candidate_risk_adjusted_beats_initial_every_seed"]
            for item in by_season.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Multi-seed evaluation report saved to: {args.output}")


if __name__ == "__main__":
    main()
