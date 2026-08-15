"""Benchmark the experimental CUDA full-season stage suite.

This includes draft, weekly projections/lineups, waivers, one-for-one trades,
standings, and playoffs. It remains separate from the production simulator
until parity reports are accepted.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from gpu_sim.full_season import (
    CudaSeasonState,
    create_synthetic_season_state,
    run_full_cuda_season,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--players", type=int, default=256)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--progress-seconds", type=float, default=10.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("reports/full_cuda_suite.json"))
    return parser.parse_args()


def clone_state(template: CudaSeasonState) -> CudaSeasonState:
    return CudaSeasonState(
        draft_projections=template.draft_projections,
        weekly_projections=template.weekly_projections,
        weekly_actual_points=template.weekly_actual_points,
        positions=template.positions,
        team_count=template.team_count,
        roster_size=template.roster_size,
    )


def main() -> None:
    args = parse_args()
    if args.duration_seconds <= 0 or args.progress_seconds <= 0:
        raise SystemExit("Duration and progress intervals must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    template = create_synthetic_season_state(
        scenarios=args.scenarios,
        players=args.players,
        device=device,
    )
    if template.device.type == "cuda":
        torch.cuda.synchronize(template.device)
    start = perf_counter()
    next_progress = args.progress_seconds
    completed = 0
    while True:
        run_full_cuda_season(clone_state(template))
        if template.device.type == "cuda":
            torch.cuda.synchronize(template.device)
        completed += 1
        elapsed = perf_counter() - start
        if elapsed >= next_progress:
            print(
                f"[Progress] seasons={completed} elapsed={elapsed / 60.0:.1f}m",
                flush=True,
            )
            next_progress += args.progress_seconds
        if elapsed >= args.duration_seconds:
            break
    elapsed = perf_counter() - start
    seasons_per_hour = completed / (elapsed / 3600.0)
    result = {
        "device": str(template.device),
        "scenarios": args.scenarios,
        "players": args.players,
        "elapsed_seconds": round(elapsed, 6),
        "duration_seconds_requested": args.duration_seconds,
        "completed_full_seasons": completed,
        "full_seasons_per_hour": round(seasons_per_hour, 2),
        "scenario_seasons_per_hour": round(seasons_per_hour * args.scenarios, 2),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Benchmark report saved to: {args.output}")


if __name__ == "__main__":
    main()
