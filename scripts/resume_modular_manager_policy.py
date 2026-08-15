"""Resume modular self-play from a complete generation-boundary state checkpoint."""

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter

from evolution.genome import DraftStrategyGenome, create_random_genome
from evolution.modular_policy_training import (
    load_modular_training_state,
    train_modular_policy_self_play,
)
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES
from fantasy_engine.weekly_data import load_weekly_performances
from models.modular_manager_policy import save_modular_policy_network
from models.transaction_value import load_transaction_value_model
from scripts.train_modular_manager_policy import (
    create_final_evaluation_callback,
    create_generation_callback,
    create_state_callback,
    create_training_league,
    write_json,
)


def load_transaction_genome() -> DraftStrategyGenome:
    path = Path("data/evolution/best_full_season_2021_genome.json")
    if path.exists():
        return DraftStrategyGenome.from_json(path.read_text(encoding="utf-8"))
    return create_random_genome(seed=2021)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-checkpoint", type=Path, required=True)
    parser.add_argument("--additional-generations", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/modular_manager_policy.pt"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/modular_resume_report.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/models/modular_policy_checkpoints"),
    )
    parser.add_argument("--transaction-value-output", type=Path, default=None)
    parser.add_argument(
        "--skip-final-evaluation",
        action="store_true",
        help="Skip the expensive all-season candidate audit for non-final segments.",
    )
    return parser.parse_args()


def load_resume_report(
    report_path: Path,
    state_checkpoint: Path,
    transaction_mode: str,
    transaction_value_output: Path,
    state_completed_generations: int,
    additional_generations: int,
    population_size: int,
) -> dict:
    """Preserve generation evidence when a segment is retried after interruption."""

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.setdefault("generations", [])
        report.setdefault("stages", {})
        report["resume_attempts"] = int(report.get("resume_attempts", 0)) + 1
    else:
        report = {
            "started_at": datetime.now().astimezone().isoformat(),
            "stages": {},
            "generations": [],
        }

    report.update(
        {
            "status": "running",
            "updated_at": datetime.now().astimezone().isoformat(),
            "resumed_from": str(state_checkpoint),
            "configuration": {
                "population": population_size,
                "additional_generations": additional_generations,
                "completed_generations_before_resume": state_completed_generations,
                "target_generations_after_resume": (
                    state_completed_generations + additional_generations
                ),
                "transaction_mode": transaction_mode,
                "transaction_value_output": str(transaction_value_output),
            },
        }
    )
    # A new attempt must replace the old final selection, while retaining every
    # generation callback already written before the interruption.
    report.pop("final_evaluation", None)
    report.pop("selected_checkpoint_path", None)
    return report


def main() -> None:
    args = parse_args()
    if args.additional_generations < 1:
        raise ValueError("additional-generations must be at least one.")

    state = load_modular_training_state(args.state_checkpoint)
    metadata = state.metadata
    start_season = int(metadata["start_season"])
    end_season = int(metadata["end_season"])
    population_size = len(state.population)
    selection_count = min(int(metadata.get("selection", 3)), population_size)
    transaction_mode = str(metadata.get("transaction_mode", "genome"))
    transaction_value_output = args.transaction_value_output or Path(
        str(metadata.get("transaction_value_output", "data/models/transaction_value_model.pt"))
    )
    transaction_value_model = None
    if transaction_mode in ("neural", "hybrid") and transaction_value_output.exists():
        transaction_value_model = load_transaction_value_model(transaction_value_output)
    elif transaction_mode in ("neural", "hybrid"):
        transaction_mode = "genome"

    seasons = list(range(start_season, end_season + 1))
    scenarios = [
        (
            create_training_league(season),
            load_weekly_performances(season, include_special_teams=True),
        )
        for season in seasons
    ]
    run_started = perf_counter()
    report = load_resume_report(
        report_path=args.report,
        state_checkpoint=args.state_checkpoint,
        transaction_mode=transaction_mode,
        transaction_value_output=transaction_value_output,
        state_completed_generations=state.completed_generations,
        additional_generations=args.additional_generations,
        population_size=population_size,
    )
    report["configuration"].update(
        {
            "start_season": start_season,
            "end_season": end_season,
        }
    )
    write_json(args.report, report)
    print(
        f"Resuming modular training at generation {state.completed_generations}; "
        f"targeting {state.completed_generations + args.additional_generations}",
        flush=True,
    )
    print(f"State checkpoint: {args.state_checkpoint}", flush=True)
    print(f"Report: {args.report}", flush=True)

    try:
        trained_model, history = train_modular_policy_self_play(
            initial_policy=state.best_policy,
            scenarios=scenarios,
            transaction_genome=load_transaction_genome(),
            population_size=population_size,
            generations=args.additional_generations,
            selection_count=selection_count,
            mutation_strength=float(metadata.get("mutation_strength", 0.01)),
            final_mutation_strength=metadata.get("final_mutation_strength"),
            risk_penalty=float(metadata.get("risk_penalty", 0.15)),
            elite_count=int(metadata.get("elite_count", 1)),
            draft_exploration_rate=float(metadata.get("draft_exploration_rate", 0.08)),
            draft_exploration_top_k=int(metadata.get("draft_exploration_top_k", 8)),
            diversity_floor=float(metadata.get("diversity_floor", 0.01)),
            diversity_mutation_boost=float(metadata.get("diversity_mutation_boost", 2.0)),
            baseline_relative_weight=float(metadata.get("baseline_relative_weight", 0.25)),
            immigrant_fraction=float(metadata.get("immigrant_fraction", 0.10)),
            transaction_ablation=bool(metadata.get("transaction_ablation", False)),
            transaction_mode=transaction_mode,
            transaction_value_model=transaction_value_model,
            seed=int(metadata.get("seed", 1)),
            rounds=int(metadata.get("rounds", 16)),
            evaluation_workers=int(metadata.get("evaluation_workers", 8)),
            lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
            scenarios_per_generation=(
                int(metadata["scenarios_per_generation"])
                if int(metadata.get("scenarios_per_generation", 0)) > 0
                else None
            ),
            full_evaluation_interval=int(metadata.get("full_evaluation_every", 0)),
            anchor_scenarios_per_generation=int(metadata.get("anchor_scenarios_per_generation", 4)),
            final_selection_count=int(metadata.get("final_selection_count", 8)),
            candidate_archive_size=int(metadata.get("candidate_archive_size", 64)),
            resume_state=state,
            total_generations=state.completed_generations + args.additional_generations,
            generation_callback=create_generation_callback(
                report,
                args.report,
                args.checkpoint_dir,
            ),
            final_evaluation_callback=create_final_evaluation_callback(
                report,
                args.report,
                args.checkpoint_dir,
            ) if not args.skip_final_evaluation else None,
            run_final_evaluation=not args.skip_final_evaluation,
            state_callback=create_state_callback(
                report=report,
                report_path=args.report,
                state_path=args.state_checkpoint,
                metadata=metadata,
            ),
        )
        model_path = save_modular_policy_network(trained_model, args.output)
        report["status"] = "completed"
        report["history"] = history
        report["model_path"] = str(model_path)
        report["elapsed_seconds"] = round(perf_counter() - run_started, 2)
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report["updated_at"] = report["finished_at"]
        write_json(args.report, report)
        print(f"Resumed modular training complete; policy saved to {model_path}", flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        report["elapsed_seconds"] = round(perf_counter() - run_started, 2)
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report["updated_at"] = report["finished_at"]
        write_json(args.report, report)
        raise


if __name__ == "__main__":
    main()
