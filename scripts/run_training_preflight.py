"""Run all fast correctness gates before a long manager training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evolution.preflight import run_training_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/training_preflight.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_season < args.start_season:
        raise SystemExit("end-season must be >= start-season")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    result = run_training_preflight(
        tuple(range(args.start_season, args.end_season + 1)),
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Training preflight approved: {result.approved}")
    print(f"Data seasons checked: {len(result.manifests)}")
    print(f"Pretraining loss: {result.pretraining_loss:.6f}")
    print(f"Report saved to: {args.output}")
    if not result.approved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
