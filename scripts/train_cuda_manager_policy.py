"""Train the modular manager policy with the CUDA full-season simulator."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch

from gpu_sim.historical_adapter import create_historical_cuda_inputs
from gpu_sim.policy_training import (
    CudaGenerationMetrics,
    save_cuda_policy_checkpoint,
    save_cuda_training_state,
    train_cuda_policy_population,
)
from models.modular_manager_policy import (
    ModularManagerPolicyNetwork,
    load_modular_policy_network,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--selection", type=int, default=4)
    parser.add_argument("--scenario-repeats", type=int, default=8)
    parser.add_argument("--projection-noise", type=float, default=0.015)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--mutation-strength", type=float, default=0.02)
    parser.add_argument("--final-mutation-strength", type=float, default=0.005)
    parser.add_argument("--draft-anchor-weight", type=float, default=0.20)
    parser.add_argument("--risk-penalty", type=float, default=0.10)
    parser.add_argument(
        "--compile-policy",
        action="store_true",
        help="Use torch.compile reduce-overhead for repeated CUDA policy forwards.",
    )
    parser.add_argument(
        "--disable-population-batching",
        action="store_true",
        help="Use sequential policy evaluation for debugging or CPU parity.",
    )
    parser.add_argument(
        "--batched-policy-heads",
        action="store_true",
        help=(
            "Use the parity-tested flattened CUDA population route for all manager heads. "
            "Without this flag, exact per-policy head evaluation remains the default."
        ),
    )
    parser.add_argument(
        "--holdout-season",
        type=int,
        default=2025,
        help="Chronological unseen season used only for final audit; use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--players", type=int, default=256)
    parser.add_argument(
        "--initial-policy",
        type=Path,
        default=Path("data/models/modular_manager_policy.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/models/cuda_manager_policy.pt"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/models/cuda_manager_training_state.pt"),
        help="Full-population checkpoint written after every generation.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume the population checkpoint written by --checkpoint.",
    )
    parser.add_argument("--report", type=Path, default=Path("reports/cuda_manager_training.json"))
    parser.add_argument(
        "--disable-transactions",
        action="store_true",
        help="Disable CUDA waiver/trade stages for a draft-only ablation.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA-capable PyTorch device is available.")
    return torch.device(requested)


def load_historical_states(args: argparse.Namespace, device: torch.device):
    seasons = list(range(args.start_season, args.end_season + 1))

    def load(season: int):
        return create_historical_cuda_inputs(
            season=season,
            players=args.players,
            device=device,
        ).state

    workers = max(1, min(args.loader_workers, len(seasons)))
    if workers == 1:
        return [load(season) for season in seasons]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(load, seasons))


def load_holdout_state(args: argparse.Namespace, device: torch.device):
    if args.holdout_season <= 0:
        return None
    return create_historical_cuda_inputs(
        season=args.holdout_season,
        players=args.players,
        device=device,
    ).state


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    initial_policy = (
        load_modular_policy_network(args.initial_policy)
        if args.initial_policy.exists()
        else ModularManagerPolicyNetwork()
    ).to(device)
    print(f"CUDA manager training device: {device}", flush=True)
    print(
        f"Seasons: {args.start_season}-{args.end_season} | population={args.population} "
        f"generations={args.generations} repeats={args.scenario_repeats}",
        flush=True,
    )
    print(
        "Population routing: "
        + (
            "flattened CUDA policy heads"
            if args.batched_policy_heads
            else "exact per-policy heads"
        ),
        flush=True,
    )
    print("Loading historical CUDA states...", flush=True)
    states = load_historical_states(args, device)
    holdout_state = load_holdout_state(args, device)
    resume_state = None
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        print(
            f"Resuming after generation {resume_state['generation']} from {args.resume}",
            flush=True,
        )
    report = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "configuration": vars(args) | {"device": str(device)},
        "generations": [],
        "holdout": None,
        "resumed_from": str(args.resume) if args.resume is not None else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    def on_generation(
        metrics: CudaGenerationMetrics,
        best_policy: ModularManagerPolicyNetwork,
    ) -> None:
        checkpoint = save_cuda_policy_checkpoint(
            best_policy,
            args.output,
            [*metrics_history, metrics],
        )
        record = metrics.to_dict() | {"checkpoint": str(checkpoint)}
        metrics_history.append(metrics)
        report["generations"].append(record)
        report["updated_at"] = datetime.now().astimezone().isoformat()
        args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(
            f"[CUDA generation {metrics.generation}/{metrics.generations}] "
            f"best={metrics.best_fitness:.2f} avg={metrics.average_fitness:.2f} "
            f"risk_adj={metrics.best_risk_adjusted_fitness:.2f} "
            f"std={metrics.best_fitness_stddev:.2f} "
            f"wins={metrics.best_wins:.2f} playoffs={metrics.best_playoff_rate:.1%} "
            f"championships={metrics.best_championship_rate:.1%} "
            f"GPH={metrics.generations_per_hour:.2f} "
            f"elapsed={metrics.elapsed_seconds / 3600:.2f}h",
            flush=True,
        )

    metrics_history: list[CudaGenerationMetrics] = []
    if resume_state is not None:
        metrics_history = [CudaGenerationMetrics(**item) for item in resume_state["metrics"]]
    best_policy, metrics_history = train_cuda_policy_population(
        initial_policy=initial_policy,
        historical_states=states,
        population_size=args.population,
        generations=args.generations,
        selection_count=args.selection,
        mutation_strength=args.mutation_strength,
        final_mutation_strength=args.final_mutation_strength,
        scenario_repeats=args.scenario_repeats,
        projection_noise=args.projection_noise,
        enable_transactions=not args.disable_transactions,
        seed=args.seed,
        draft_anchor_weight=args.draft_anchor_weight,
        risk_penalty=args.risk_penalty,
        compile_policy=args.compile_policy,
        batch_population=not args.disable_population_batching,
        exact_policy_head_parity=not args.batched_policy_heads,
        resume_state=resume_state,
        generation_callback=on_generation,
        checkpoint_callback=lambda generation, population, best_policy, metrics, rng: (
            save_cuda_training_state(
                args.checkpoint,
                generation=generation,
                population=population,
                best_policy=best_policy,
                metrics=metrics,
                rng_state=rng.getstate(),
            )
        ),
    )
    # The callback writes the full resumable population checkpoint each
    # generation; this final write records the terminal best-policy artifact.
    save_cuda_policy_checkpoint(best_policy, args.output, metrics_history)
    report["status"] = "complete"
    report["completed_at"] = datetime.now().astimezone().isoformat()
    report["generations"] = [metric.to_dict() for metric in metrics_history]
    report["output"] = str(args.output)
    if holdout_state is not None:
        from gpu_sim.policy_training import evaluate_cuda_policy

        holdout = evaluate_cuda_policy(
            best_policy,
            [holdout_state],
            scenario_repeats=max(args.scenario_repeats, 8),
            projection_noise=args.projection_noise,
            enable_transactions=not args.disable_transactions,
            seed=args.seed + 900_000,
            draft_anchor_weight=args.draft_anchor_weight,
            risk_penalty=args.risk_penalty,
            compile_policy=args.compile_policy,
        )
        report["holdout"] = {
            "season": args.holdout_season,
            "fitness": holdout.fitness,
            "fitness_stddev": holdout.fitness_stddev,
            "risk_adjusted_fitness": holdout.risk_adjusted_fitness,
            "wins": holdout.wins,
            "points_for": holdout.points_for,
            "playoff_rate": holdout.playoff_rate,
            "championship_rate": holdout.championship_rate,
            "elapsed_seconds": holdout.elapsed_seconds,
        }
        print(
            f"Holdout {args.holdout_season}: fitness={holdout.fitness:.2f} "
            f"wins={holdout.wins:.2f} playoffs={holdout.playoff_rate:.1%} "
            f"championships={holdout.championship_rate:.1%}",
            flush=True,
        )
    args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"CUDA manager policy saved to: {args.output}", flush=True)
    print(f"Training report saved to: {args.report}", flush=True)


if __name__ == "__main__":
    main()
