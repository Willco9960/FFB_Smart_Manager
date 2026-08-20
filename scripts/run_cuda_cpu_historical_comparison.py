"""Run a multi-season CPU/CUDA historical comparison with live progress."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

# Keep direct ``python scripts/...py`` execution equivalent to module execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from gpu_sim.historical_adapter import create_historical_cuda_inputs
from scripts.compare_cpu_cuda_historical_season import summarize_cpu, summarize_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--players", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--transactions", action="store_true")
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cpu_cuda_historical_comparison.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_season < args.start_season:
        raise SystemExit("end-season must be at least start-season.")
    if args.progress_seconds <= 0:
        raise SystemExit("progress-seconds must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    seasons = list(range(args.start_season, args.end_season + 1))
    started = perf_counter()
    next_progress = args.progress_seconds
    results = []
    for index, season in enumerate(seasons, start=1):
        inputs = create_historical_cuda_inputs(
            season=season,
            players=args.players,
            device=args.device,
        )
        cuda_started = perf_counter()
        cuda = summarize_cuda(inputs, args.transactions)
        if inputs.state.device.type == "cuda":
            torch.cuda.synchronize(inputs.state.device)
        cuda_elapsed = perf_counter() - cuda_started
        cpu_started = perf_counter()
        cpu = summarize_cpu(
            inputs,
            args.transactions,
            roster_indices=inputs.state.rosters[0].detach().cpu().tolist(),
        )
        cpu_elapsed = perf_counter() - cpu_started
        results.append(
            {
                "season": season,
                "cpu_elapsed_seconds": round(cpu_elapsed, 4),
                "cuda_elapsed_seconds": round(cuda_elapsed, 4),
                "cpu": cpu,
                "cuda": cuda,
                "exact_standings_match": cpu["standings"] == cuda["standings"],
                "exact_champion_match": cpu["champion"] == cuda["champion"],
            }
        )
        elapsed = perf_counter() - started
        if elapsed >= next_progress:
            print(
                f"[Progress] season {index}/{len(seasons)} ({season}) "
                f"elapsed={elapsed / 60.0:.1f}m",
                flush=True,
            )
            next_progress += args.progress_seconds

    total_elapsed = perf_counter() - started
    report = {
        "start_season": args.start_season,
        "end_season": args.end_season,
        "players": args.players,
        "device": str(args.device),
        "transactions": args.transactions,
        "comparison_mode": ("transaction_outcome_delta" if args.transactions else "exact_parity"),
        "seasons": results,
        "exact_standings_match_count": sum(item["exact_standings_match"] for item in results),
        "exact_champion_match_count": sum(item["exact_champion_match"] for item in results),
        "elapsed_seconds": round(total_elapsed, 4),
        "seasons_per_hour": round(len(seasons) / (total_elapsed / 3600.0), 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Comparison report saved to: {args.output}")


if __name__ == "__main__":
    main()
