"""Print the durable status of a modular vacation-training run."""

import argparse
import json
from pathlib import Path

from evolution.modular_policy_training import load_modular_training_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id")
    group.add_argument("--manifest", type=Path)
    return parser.parse_args()


def resolve_manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest is not None:
        return args.manifest
    return Path("data/models/vacation_runs") / args.run_id / "manifest.json"


def main() -> None:
    manifest_path = resolve_manifest_path(parse_args())
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Run: {manifest.get('run_id', manifest_path.parent.name)}")
    print(f"Status: {manifest.get('status', 'unknown')}")
    print(f"Segments: {manifest.get('completed_segments', 0)} / {manifest.get('segments', '?')}")
    print(f"Current segment: {manifest.get('current_segment', 'none')}")
    print(f"Updated: {manifest.get('updated_at', 'unknown')}")
    print(f"Manifest: {manifest_path}")

    state_path = Path(str(manifest["state_checkpoint"]))
    if not state_path.exists():
        print("State checkpoint: not created yet")
        return

    state = load_modular_training_state(state_path)
    print(f"Generations: {state.completed_generations} / {state.target_generations}")
    print(f"Best risk-adjusted score: {state.best_risk_adjusted_score:.2f}")
    print(f"Best generation: {state.best_generation}")
    print(f"Elapsed training time: {state.elapsed_seconds / 3600:.2f}h")
    print(f"State checkpoint: {state_path}")


if __name__ == "__main__":
    main()
