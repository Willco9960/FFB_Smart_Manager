"""Compare the isolated tensorized backend on CPU and CUDA.

The report is a kernel comparison, not a claim about full-season simulator
throughput. Both devices receive the same generated tensors and seeds.
"""

import argparse
import json
from pathlib import Path

import torch

from gpu_sim.tensor_state import create_synthetic_scenario_batch
from gpu_sim.tensorized_draft import benchmark_tensorized_draft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=4096)
    parser.add_argument("--players", type=int, default=512)
    parser.add_argument("--teams", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tensorized_backend_comparison.json"),
    )
    return parser.parse_args()


def create_inputs(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = create_synthetic_scenario_batch(args.scenarios, args.players)
    return batch.projected_points, batch.actual_points, batch.positions


def main() -> None:
    args = parse_args()
    projected, actual, positions = create_inputs(args)
    results = {
        "cpu": benchmark_tensorized_draft(
            projected,
            actual,
            positions,
            team_count=args.teams,
            rounds=args.rounds,
            repeats=args.repeats,
        )
    }

    if torch.cuda.is_available():
        results["cuda"] = benchmark_tensorized_draft(
            projected.cuda(),
            actual.cuda(),
            positions.cuda(),
            team_count=args.teams,
            rounds=args.rounds,
            repeats=args.repeats,
        )
        results["cuda_speedup_vs_cpu"] = round(
            results["cuda"]["batch_runs_per_hour"]
            / results["cpu"]["batch_runs_per_hour"],
            3,
        )
    else:
        results["cuda"] = None
        results["cuda_speedup_vs_cpu"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Comparison report saved to: {args.output}")


if __name__ == "__main__":
    main()
