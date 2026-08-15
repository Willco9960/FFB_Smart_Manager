"""Run bounded modular-training segments with durable progress manifests."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--segments", type=int, default=10)
    parser.add_argument("--generations-per-segment", type=int, default=10)
    parser.add_argument("--start-season", type=int, default=2001)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--selection", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--offline-epochs", type=int, default=50)
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=8,
        help="Parallel historical-scenario workers; 1 disables multiprocessing.",
    )
    parser.add_argument("--transaction-value-epochs", type=int, default=100)
    parser.add_argument(
        "--transaction-mode",
        choices=("genome", "neural", "hybrid"),
        default="genome",
        help="Transaction strategy used during self-play; genome is the validated default.",
    )
    parser.add_argument(
        "--transaction-ablation",
        action="store_true",
        help="Evaluate transaction arms at final selection time.",
    )
    parser.add_argument(
        "--collect-season-replay",
        action="store_true",
        help="Collect replay and train the transaction-value model for neural/hybrid modes.",
    )
    parser.add_argument("--scenarios-per-generation", type=int, default=12)
    parser.add_argument("--full-evaluation-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore an existing run manifest and start a new run with this run ID.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def run_command(command: list[str], root: Path, dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=root, check=True)


def main() -> None:
    args = parse_args()
    if args.segments < 1:
        raise ValueError("segments must be at least one.")
    if args.generations_per_segment < 1:
        raise ValueError("generations-per-segment must be at least one.")
    if args.selection > args.population:
        raise ValueError("selection cannot exceed population.")

    root = Path(__file__).resolve().parents[1]
    run_id = args.run_id or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_root = root / "data" / "models" / "vacation_runs" / run_id
    report_root = root / "reports" / "vacation" / run_id
    log_root = root / "logs" / "vacation" / run_id
    state_path = run_root / "training_state.pt"
    checkpoint_dir = run_root / "generation_checkpoints"
    transaction_value_path = run_root / "transaction_value_model.pt"
    manifest_path = run_root / "manifest.json"
    log_root.mkdir(parents=True, exist_ok=True)
    existing_manifest: dict[str, object] | None = None
    if manifest_path.exists() and not args.force_restart:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("status") == "completed":
            raise RuntimeError(
                f"Run {run_id} is already complete. Use a new --run-id or --force-restart."
            )

    completed_segments = int((existing_manifest or {}).get("completed_segments", 0))
    if completed_segments >= args.segments:
        raise RuntimeError(
            "The existing manifest has completed at least the requested number of segments. "
            "Use a larger --segments value, a new --run-id, or --force-restart."
        )

    manifest: dict[str, object] = existing_manifest or {
        "status": "planned" if args.dry_run else "running",
        "run_id": run_id,
        "started_at": datetime.now().astimezone().isoformat(),
        "segments": args.segments,
        "generations_per_segment": args.generations_per_segment,
        "completed_segments": 0,
        "state_checkpoint": str(state_path),
        "report_root": str(report_root),
        "log_root": str(log_root),
        "segment_reports": [],
        "commands": [],
        "transaction_mode": args.transaction_mode,
        "transaction_ablation": args.transaction_ablation,
        "collect_season_replay": args.collect_season_replay,
        "evaluation_workers": args.evaluation_workers,
    }
    manifest.update(
        {
            "segments": args.segments,
            "generations_per_segment": args.generations_per_segment,
            "state_checkpoint": str(state_path),
            "report_root": str(report_root),
            "log_root": str(log_root),
            "status": "planned" if args.dry_run else "running",
            "transaction_mode": args.transaction_mode,
            "transaction_ablation": args.transaction_ablation,
            "collect_season_replay": args.collect_season_replay,
            "evaluation_workers": args.evaluation_workers,
        }
    )
    manifest.setdefault("segment_reports", [])
    manifest.setdefault("commands", [])
    write_manifest(manifest_path, manifest)

    state_completed_generations = 0
    if state_path.exists() and not args.force_restart and not args.dry_run:
        from evolution.modular_policy_training import load_modular_training_state

        state_completed_generations = load_modular_training_state(state_path).completed_generations

    common = [
        sys.executable,
        "-u",
        "-m",
        "scripts.train_modular_manager_policy",
        "--start-season",
        str(args.start_season),
        "--end-season",
        str(args.end_season),
        "--population",
        str(args.population),
        "--selection",
        str(args.selection),
        "--epochs",
        str(args.epochs),
        "--offline-epochs",
        str(args.offline_epochs),
        "--evaluation-workers",
        str(args.evaluation_workers),
        "--transaction-mode",
        args.transaction_mode,
        "--transaction-value-epochs",
        str(args.transaction_value_epochs),
        "--scenarios-per-generation",
        str(args.scenarios_per_generation),
        "--full-evaluation-every",
        str(args.full_evaluation_every),
        "--seed",
        str(args.seed),
        "--transaction-value-output",
        str(transaction_value_path),
        "--state-checkpoint",
        str(state_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if args.transaction_ablation:
        common.append("--transaction-ablation")
    if args.collect_season_replay:
        common.append("--collect-season-replay")

    try:
        start_segment = completed_segments + 1
        for segment_number in range(start_segment, args.segments + 1):
            report_path = report_root / f"segment_{segment_number:03d}.json"
            output_path = run_root / f"policy_segment_{segment_number:03d}.pt"
            segment_target_generation = segment_number * args.generations_per_segment
            if not args.dry_run and state_completed_generations >= segment_target_generation:
                manifest["completed_segments"] = segment_number
                completed_segments = segment_number
                manifest["segment_reports"].append(str(report_path))
                manifest["updated_at"] = datetime.now().astimezone().isoformat()
                write_manifest(manifest_path, manifest)
                continue

            has_prior_state = state_completed_generations > 0
            if segment_number == 1 and not has_prior_state:
                command = common + [
                    "--generations",
                    str(args.generations_per_segment),
                    "--output",
                    str(output_path),
                    "--report",
                    str(report_path),
                ]
            else:
                remaining_generations = (
                    args.generations_per_segment
                    if args.dry_run
                    else max(1, segment_target_generation - state_completed_generations)
                )
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.resume_modular_manager_policy",
                    "--state-checkpoint",
                    str(state_path),
                    "--additional-generations",
                    str(remaining_generations),
                    "--output",
                    str(output_path),
                    "--report",
                    str(report_path),
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--transaction-value-output",
                    str(transaction_value_path),
                ]

            # Candidate audits are only needed once, after the final segment.
            # Skipping them between segments preserves checkpoints and can save
            # substantial wall time on long resumable runs.
            if segment_number < args.segments:
                command.append("--skip-final-evaluation")

            manifest["commands"].append(" ".join(command))
            manifest["current_segment"] = segment_number
            manifest["updated_at"] = datetime.now().astimezone().isoformat()
            write_manifest(manifest_path, manifest)
            print(
                f"[Vacation run {run_id}] starting segment {segment_number}/{args.segments}",
                flush=True,
            )
            run_command(command, root, args.dry_run)
            if not args.dry_run:
                manifest["completed_segments"] = segment_number
                completed_segments = segment_number
                state_completed_generations = segment_target_generation
                manifest["segment_reports"].append(str(report_path))
            manifest["updated_at"] = datetime.now().astimezone().isoformat()
            write_manifest(manifest_path, manifest)

        manifest["status"] = "planned" if args.dry_run else "completed"
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        write_manifest(manifest_path, manifest)
        print(f"Vacation training run {run_id} complete", flush=True)
        print(f"Manifest: {manifest_path}", flush=True)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        manifest["failed_at"] = datetime.now().astimezone().isoformat()
        write_manifest(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
