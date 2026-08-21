"""Run CPU-authoritative CPU/CUDA historical parity across seasons."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from gpu_sim.historical_adapter import create_historical_cuda_inputs
from scripts.compare_cpu_cuda_historical_season import (
    _public_summary,
    _sorted_reward_signature,
    summarize_cpu,
    summarize_cuda,
)


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


def _compare_season(season: int, players: int, device: str, transactions: bool) -> dict:
    cpu_inputs = create_historical_cuda_inputs(
        season=season,
        players=players,
        device="cpu",
    )
    cpu_started = perf_counter()
    cpu = summarize_cpu(cpu_inputs, transactions, include_trace=transactions)
    cpu_elapsed = perf_counter() - cpu_started

    cuda_inputs = create_historical_cuda_inputs(
        season=season,
        players=players,
        device=device,
    )
    events = [[event for event in cpu["transaction_event_objects"]]] if transactions else None
    cuda_started = perf_counter()
    cuda = summarize_cuda(
        cuda_inputs,
        transactions,
        initial_rosters=cpu.get("roster_indices"),
        canonical_transaction_events=events,
    )
    if cuda_inputs.state.device.type == "cuda":
        torch.cuda.synchronize(cuda_inputs.state.device)
    cuda_elapsed = perf_counter() - cuda_started

    cpu_events = cpu.get("transaction_events", [])
    cuda_events = cuda.get("transaction_events", [[]])[0]
    action_exact = cpu_events == cuda_events if transactions else True
    state_exact = (
        action_exact
        and all(
            left["pre_state_digest"] == right["pre_state_digest"]
            and left["post_state_digest"] == right["post_state_digest"]
            for left, right in zip(cpu_events, cuda_events, strict=True)
        )
        if transactions
        else True
    )
    first_divergence = None
    if transactions and not action_exact:
        for index, (left, right) in enumerate(zip(cpu_events, cuda_events, strict=False)):
            if left != right:
                first_divergence = {
                    "sequence_index": index,
                    "cpu_event": left,
                    "cuda_event": right,
                }
                break
        if first_divergence is None and len(cpu_events) != len(cuda_events):
            first_divergence = {
                "sequence_index": min(len(cpu_events), len(cuda_events)),
                "reason": "event_count_mismatch",
                "cpu_count": len(cpu_events),
                "cuda_count": len(cuda_events),
            }

    return {
        "season": season,
        "cpu_elapsed_seconds": round(cpu_elapsed, 4),
        "cuda_elapsed_seconds": round(cuda_elapsed, 4),
        "cpu": _public_summary(cpu),
        "cuda": cuda,
        "exact_standings_match": cpu["standings"] == cuda["standings"],
        "exact_champion_match": cpu["champion"] == cuda["champion"],
        "exact_weekly_score_match": cpu["weekly_scores"] == cuda["weekly_scores"],
        "transaction_actions_exact": action_exact,
        "transaction_state_exact": state_exact,
        "transaction_reward_exact": (
            _sorted_reward_signature(cpu)
            == sorted(
                cuda.get("transaction_reward_signature", [[]])[0],
                key=lambda item: (
                    item["week"],
                    item["transaction_type"],
                    item["team_name"],
                    item["incoming_points"],
                    item["outgoing_points"],
                    item["net_points"],
                ),
            )
            if transactions
            else True
        ),
        "transaction_event_count": len(cpu_events),
        "first_divergence": first_divergence,
    }


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
        result = _compare_season(season, args.players, args.device, args.transactions)
        results.append(result)
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
        "comparison_mode": "cpu_trace_replay" if args.transactions else "exact_parity",
        "seasons": results,
        "exact_standings_match_count": sum(item["exact_standings_match"] for item in results),
        "exact_champion_match_count": sum(item["exact_champion_match"] for item in results),
        "exact_weekly_score_match_count": sum(item["exact_weekly_score_match"] for item in results),
        "transaction_actions_exact_count": sum(
            item["transaction_actions_exact"] for item in results
        ),
        "transaction_state_exact_count": sum(item["transaction_state_exact"] for item in results),
        "transaction_reward_exact_count": sum(item["transaction_reward_exact"] for item in results),
        "elapsed_seconds": round(total_elapsed, 4),
        "seasons_per_hour": round(len(seasons) / (total_elapsed / 3600.0), 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "seasons"}, indent=2))
    print(f"Comparison report saved to: {args.output}")


if __name__ == "__main__":
    main()
