"""Benchmark the isolated tensorized CUDA experiment.

This command never calls the production league simulator. It measures the
batched draft/lineup kernel and writes a comparable JSON report.
"""

import argparse
import json
from pathlib import Path

import torch

from gpu_sim.tensorized_draft import (
    benchmark_tensorized_draft,
    benchmark_tensorized_for_duration,
)
from gpu_sim.tensor_state import create_synthetic_scenario_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=1024)
    parser.add_argument("--players", type=int, default=256)
    parser.add_argument("--teams", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Run continuously for this duration instead of a fixed repeat count.",
    )
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tensorized_cuda_benchmark.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false.")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    batch = create_synthetic_scenario_batch(
        args.scenarios,
        args.players,
        device=device,
    )
    projected = batch.projected_points
    actual = batch.actual_points
    positions = batch.positions

    if args.duration_seconds is None:
        result = benchmark_tensorized_draft(
            projected,
            actual,
            positions,
            team_count=args.teams,
            rounds=args.rounds,
            repeats=args.repeats,
            profile_stages=args.profile_stages,
        )
    else:

        def report_progress(completed_batches: int, elapsed_seconds: float) -> None:
            print(
                f"[Progress] batches={completed_batches} "
                f"elapsed={elapsed_seconds / 60.0:.1f}m",
                flush=True,
            )

        result = benchmark_tensorized_for_duration(
            projected,
            actual,
            positions,
            team_count=args.teams,
            rounds=args.rounds,
            duration_seconds=args.duration_seconds,
            progress_seconds=args.progress_seconds,
            progress_callback=report_progress,
        )
    result["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Benchmark report saved to: {args.output}")


if __name__ == "__main__":
    main()
