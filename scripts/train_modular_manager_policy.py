"""Warm-start and self-play train the modular manager policy."""

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

from agents.genome_draft_agent import GenomeDraftAgent
from agents.trade_agent import GenomeTradeAgent
from agents.waiver_agent import GenomeWaiverAgent
from evolution.genome import create_random_genome
from evolution.modular_behavior_cloning import (
    ModularImitationExample,
)
from evolution.modular_policy_training import (
    ModularGenerationMetrics,
    ModularTrainingState,
    save_modular_training_state,
    train_modular_policy_self_play,
)
from evolution.offline_replay import (
    DecisionReplayBuffer,
    DecisionReplayRecord,
    train_offline_policy,
)
from evolution.pretraining import build_manager_teacher_examples, run_manager_pretraining
from evolution.transaction_value_training import train_transaction_value_model_with_validation
from fantasy_engine.data_availability import validate_training_seasons
from fantasy_engine.draft import get_snake_draft_order, run_snake_draft
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import load_weekly_performances
from fantasy_engine.weekly_season_simulation import run_historical_regular_season
from models.modular_manager_policy import (
    ModularManagerPolicyNetwork,
    create_modular_policy_features,
    save_modular_policy_network,
)
from models.transaction_value import (
    TRANSACTION_VALUE_PATH,
    TransactionValueNetwork,
    save_transaction_value_model,
)

OUTPUT_PATH = Path("data/models/modular_manager_policy.pt")
REPORT_PATH = Path("reports/modular_training_report.json")
CHECKPOINT_DIR = Path("data/models/modular_policy_checkpoints")
STATE_CHECKPOINT_PATH = Path("data/models/modular_policy_training_state.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--selection", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--offline-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mutation-strength", type=float, default=0.01)
    parser.add_argument(
        "--final-mutation-strength",
        type=float,
        default=None,
        help="Mutation strength at the final generation; defaults to 25%% of the initial strength.",
    )
    parser.add_argument(
        "--risk-penalty",
        type=float,
        default=0.15,
        help="Penalty applied to cross-season fitness volatility during selection.",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=1,
        help="Number of globally best policies preserved into each next generation.",
    )
    parser.add_argument(
        "--draft-exploration-rate",
        type=float,
        default=0.08,
        help=(
            "Probability that a neural drafter samples among its top candidates "
            "to preserve room diversity."
        ),
    )
    parser.add_argument(
        "--draft-exploration-top-k",
        type=int,
        default=8,
        help="Candidate pool used when draft exploration triggers.",
    )
    parser.add_argument(
        "--diversity-floor",
        type=float,
        default=0.01,
        help="Increase mutation when normalized population diversity falls below this value.",
    )
    parser.add_argument(
        "--diversity-mutation-boost",
        type=float,
        default=2.0,
        help="Multiplier applied to mutation while the population is collapsing.",
    )
    parser.add_argument(
        "--baseline-relative-weight",
        type=float,
        default=0.25,
        help=(
            "Weight given to fitness relative to same-scenario baseline opponents "
            "during selection."
        ),
    )
    parser.add_argument(
        "--immigrant-fraction",
        type=float,
        default=0.10,
        help=(
            "Fraction of child policies periodically re-seeded from the warm-start "
            "policy to prevent population collapse."
        ),
    )
    parser.add_argument(
        "--transaction-ablation",
        action="store_true",
        help=(
            "Compare neural transactions with genome-baseline and disabled "
            "transaction arms during final evaluation."
        ),
    )
    parser.add_argument(
        "--transaction-mode",
        choices=("neural", "genome", "hybrid", "disabled"),
        default="hybrid",
        help="Transaction policy used during self-play and as the primary final-evaluation arm.",
    )
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument(
        "--training-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for batched behavioral/replay training; simulation remains CPU-bound.",
    )
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=8,
        help="Parallel historical-scenario workers; 1 disables multiprocessing.",
    )
    parser.add_argument(
        "--scenarios-per-generation",
        type=int,
        default=0,
        help="Optional rotating historical subset; 0 evaluates every season each generation.",
    )
    parser.add_argument(
        "--full-evaluation-every",
        type=int,
        default=0,
        help="Run all seasons every N generations when scenario sampling is enabled.",
    )
    parser.add_argument(
        "--anchor-scenarios-per-generation",
        type=int,
        default=4,
        help="Fixed seasons included in every sampled generation for comparable fitness.",
    )
    parser.add_argument(
        "--final-selection-count",
        type=int,
        default=8,
        help="Top sampled-generation candidates to compare on all seasons at the end.",
    )
    parser.add_argument(
        "--candidate-archive-size",
        type=int,
        default=64,
        help="Number of top generation candidates retained in checkpoints for final auditing.",
    )
    parser.add_argument(
        "--collect-season-replay",
        action="store_true",
        help="Collect leakage-safe lineup, waiver, and trade replay records from training seasons.",
    )
    parser.add_argument(
        "--skip-final-evaluation",
        action="store_true",
        help=(
            "Skip the expensive all-season candidate audit; useful for non-final "
            "vacation segments."
        ),
    )
    parser.add_argument(
        "--transaction-value-epochs",
        type=int,
        default=50,
        help="Epochs used to train the waiver/trade value-risk model from replay rewards.",
    )
    parser.add_argument(
        "--transaction-value-validation-seasons",
        type=int,
        default=2,
        help="Most recent seasons held out for chronological transaction-value validation.",
    )
    parser.add_argument(
        "--transaction-value-min-validation-records",
        type=int,
        default=50,
        help="Minimum holdout records required before the value model can activate.",
    )
    parser.add_argument(
        "--transaction-value-output",
        type=Path,
        default=TRANSACTION_VALUE_PATH,
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument(
        "--state-checkpoint",
        type=Path,
        default=STATE_CHECKPOINT_PATH,
        help="Atomic full training-state checkpoint written after each generation.",
    )
    parser.add_argument(
        "--overnight-profile",
        action="store_true",
        help=(
            "Use a bounded 8-9 hour configuration: population 24, 10 generations, "
            "12 rotating seasons, and a full evaluation every 5 generations."
        ),
    )
    args = parser.parse_args()
    if args.overnight_profile:
        args.population = 24
        args.generations = 10
        args.selection = 8
        args.scenarios_per_generation = 12
        args.full_evaluation_every = 5
        args.anchor_scenarios_per_generation = 6
        args.risk_penalty = 0.15
        args.draft_exploration_rate = 0.08
        args.draft_exploration_top_k = 8
        args.final_selection_count = 8
        args.baseline_relative_weight = 0.25
        args.immigrant_fraction = 0.10
    return args


def resolve_training_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA-capable PyTorch device is available.")
    return torch.device(requested)


def create_training_league(season: int) -> League:
    return League(
        name=f"Modular Training League {season}",
        teams=[Team(name=f"Modular Team {number}") for number in range(1, 11)],
        available_players=load_leakage_safe_player_pool(
            projection_season=season - 1,
            actual_season=season,
            include_special_teams=True,
        )[:250],
    )


def collect_draft_examples(
    league: League,
    teacher: GenomeDraftAgent,
    episodes: int = 2,
    rounds: int = 16,
) -> list[ModularImitationExample]:
    examples = []
    for _ in range(episodes):
        episode_league = League(
            name=league.name,
            teams=[Team(name=team.name) for team in league.teams],
            available_players=list(league.available_players),
        )
        for round_number in range(1, rounds + 1):
            for team in get_snake_draft_order(episode_league.teams, round_number):
                available = episode_league.available_players
                scores = [teacher.score_player(player, available) for player in available]
                maximum = max(max(scores), 1.0)
                examples.extend(
                    ModularImitationExample(
                        features=create_modular_policy_features(player, team, available),
                        target_score=score / maximum,
                    )
                    for player, score in zip(available, scores, strict=True)
                )
                selected = teacher.choose_player(available, team, episode_league)
                team.add_player(selected)
                available.remove(selected)
    return examples


def collect_season_replay(
    seasons: list[int],
    genome,
) -> DecisionReplayBuffer:
    replay_buffer = DecisionReplayBuffer()
    for season in seasons:
        league = create_training_league(season)
        draft_agent = GenomeDraftAgent(genome)
        team_agents = {team.name: draft_agent for team in league.teams}
        run_snake_draft(league, rounds=16, team_agents=team_agents)
        waiver_agents = {team.name: GenomeWaiverAgent(genome=genome) for team in league.teams}
        trade_agents = {team.name: GenomeTradeAgent(genome=genome) for team in league.teams}
        result = run_historical_regular_season(
            league=league,
            performances=load_weekly_performances(season, include_special_teams=True),
            waiver_agents=waiver_agents,
            trade_agents=trade_agents,
            season=season,
        )
        replay_buffer.extend(result.decision_replay_records)
    return replay_buffer


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_generation_callback(
    report: dict,
    report_path: Path,
    checkpoint_dir: Path,
):
    def on_generation(metrics: ModularGenerationMetrics, best_policy) -> None:
        checkpoint_path = checkpoint_dir / (
            f"after_generation_{metrics.generation_number:03d}_"
            f"best_from_generation_{metrics.cumulative_best_generation:03d}.pt"
        )
        save_modular_policy_network(best_policy, checkpoint_path)
        generation_record = metrics.to_dict()
        generation_record["checkpoint_path"] = str(checkpoint_path)
        report["generations"].append(generation_record)
        report["best_checkpoint_path"] = str(checkpoint_path)
        report["updated_at"] = datetime.now().astimezone().isoformat()
        write_json(report_path, report)

        print(
            f"[Generation {metrics.generation_number}/{metrics.generation_count}] "
            f"best={metrics.best_fitness:.2f} "
            f"avg={metrics.average_fitness:.2f} "
            f"median={metrics.median_fitness:.2f} "
            f"std={metrics.fitness_stddev:.2f} "
            f"wins={metrics.best_wins:.2f} "
            f"playoffs={metrics.best_playoff_rate:.1%} "
            f"championships={metrics.best_championship_rate:.1%} "
            f"baseline_avg={metrics.baseline_average_fitness} "
            f"relative={metrics.best_baseline_relative_fitness} "
            f"selection={metrics.selection_score} "
            f"diversity={metrics.policy_population_diversity:.4f} "
            f"gph={metrics.generations_per_hour:.2f} "
            f"elapsed={metrics.elapsed_seconds / 3600:.2f}h",
            flush=True,
        )

    return on_generation


def create_state_callback(
    report: dict,
    report_path: Path,
    state_path: Path,
    metadata: dict[str, object],
):
    def on_state(state: ModularTrainingState) -> None:
        state.metadata.update(metadata)
        save_modular_training_state(state, state_path)
        report["state_checkpoint_path"] = str(state_path)
        report["completed_generations"] = state.completed_generations
        report["updated_at"] = datetime.now().astimezone().isoformat()
        write_json(report_path, report)
        print(
            f"[Checkpoint] generation {state.completed_generations}/"
            f"{state.target_generations} saved to {state_path}",
            flush=True,
        )

    return on_state


def create_final_evaluation_callback(
    report: dict,
    report_path: Path,
    checkpoint_dir: Path,
):
    def on_final_evaluation(final_report: dict, selected_policy) -> None:
        selected_generation = final_report["selected_generation"]
        checkpoint_path = checkpoint_dir / (
            f"selected_full_evaluation_generation_{selected_generation:03d}.pt"
        )
        save_modular_policy_network(selected_policy, checkpoint_path)
        final_report["selected_checkpoint_path"] = str(checkpoint_path)
        report["final_evaluation"] = final_report
        report["selected_checkpoint_path"] = str(checkpoint_path)
        report["selected_transaction_mode"] = final_report["selected_transaction_mode"]
        report["updated_at"] = datetime.now().astimezone().isoformat()
        write_json(report_path, report)

        selected = next(
            candidate
            for candidate in final_report["candidates"]
            if candidate["generation_number"] == selected_generation
        )
        try:
            print(
                f"[Final full evaluation] selected generation={selected_generation} "
                f"mode={selected['recommended_transaction_mode']} "
                f"fitness={selected['full_evaluation_fitness']:.2f} "
                f"selected_score={selected['recommended_transaction_risk_adjusted_fitness']:.2f} "
                f"wins={selected['wins']:.2f} "
                f"playoffs={selected['playoff_rate']:.1%} "
                f"championships={selected['championship_rate']:.1%}",
                flush=True,
            )
        except OSError:
            # A closed terminal/pipeline must not turn a completed training run
            # into a failed run after the report and checkpoint were written.
            pass

    return on_final_evaluation


def main() -> None:
    args = parse_args()
    training_device = resolve_training_device(args.training_device)
    started_at = datetime.now().astimezone()
    run_started = perf_counter()
    seasons = list(range(args.start_season, args.end_season + 1))
    availability_seasons = sorted({value for season in seasons for value in (season - 1, season)})
    data_manifests = validate_training_seasons(availability_seasons)
    report = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "updated_at": started_at.isoformat(),
        "configuration": {
            "start_season": args.start_season,
            "end_season": args.end_season,
            "seasons": seasons,
            "population": args.population,
            "generations": args.generations,
            "selection": args.selection,
            "epochs": args.epochs,
            "offline_epochs": args.offline_epochs,
            "seed": args.seed,
            "mutation_strength": args.mutation_strength,
            "final_mutation_strength": args.final_mutation_strength,
            "risk_penalty": args.risk_penalty,
            "elite_count": args.elite_count,
            "draft_exploration_rate": args.draft_exploration_rate,
            "draft_exploration_top_k": args.draft_exploration_top_k,
            "diversity_floor": args.diversity_floor,
            "diversity_mutation_boost": args.diversity_mutation_boost,
            "baseline_relative_weight": args.baseline_relative_weight,
            "immigrant_fraction": args.immigrant_fraction,
            "transaction_ablation": args.transaction_ablation,
            "transaction_mode": args.transaction_mode,
            "rounds": args.rounds,
            "training_device": str(training_device),
            "evaluation_workers": args.evaluation_workers,
            "scenarios_per_generation": args.scenarios_per_generation,
            "full_evaluation_every": args.full_evaluation_every,
            "anchor_scenarios_per_generation": args.anchor_scenarios_per_generation,
            "final_selection_count": args.final_selection_count,
            "candidate_archive_size": args.candidate_archive_size,
            "collect_season_replay": args.collect_season_replay,
            "skip_final_evaluation": args.skip_final_evaluation,
            "transaction_value_epochs": args.transaction_value_epochs,
            "transaction_value_validation_seasons": args.transaction_value_validation_seasons,
            "transaction_value_min_validation_records": (
                args.transaction_value_min_validation_records
            ),
            "transaction_value_output": str(args.transaction_value_output),
            "overnight_profile": args.overnight_profile,
            "state_checkpoint": str(args.state_checkpoint),
        },
        "stages": {},
        "generations": [],
        "data_availability": [manifest.to_dict() for manifest in data_manifests],
    }
    write_json(args.report, report)
    print(f"Training started: {started_at.isoformat()}", flush=True)
    print(f"Seasons: {args.start_season}-{args.end_season} ({len(seasons)} scenarios)", flush=True)
    print(f"Batched training device: {training_device}", flush=True)
    if args.overnight_profile:
        print(
            "Overnight profile: population=24 generations=10 "
            "scenarios_per_generation=12 full_evaluation_every=5",
            flush=True,
        )
    if args.scenarios_per_generation > 0:
        print(
            f"Scenario rotation: {args.scenarios_per_generation} per generation; "
            f"{args.anchor_scenarios_per_generation} fixed anchors; "
            f"full evaluation every {args.full_evaluation_every or 'never'} generations",
            flush=True,
        )
    print(f"Report: {args.report}", flush=True)
    missing_context = sorted(
        {
            column
            for manifest in data_manifests
            for column in manifest.missing_optional_columns
        }
    )
    if missing_context:
        print(
            "Optional context feeds unavailable; features will remain explicitly masked: "
            f"{', '.join(missing_context)}",
            flush=True,
        )

    try:
        model = ModularManagerPolicyNetwork()
        teacher = GenomeDraftAgent(create_random_genome(seed=2021))

        stage_started = perf_counter()
        print("[Stage 1/4] Collecting behavioral examples for all policy heads...", flush=True)
        first_league = create_training_league(args.start_season)
        examples = build_manager_teacher_examples(first_league, episodes=1, rounds=args.rounds)
        pretraining_result = run_manager_pretraining(
            model,
            examples,
            behavior_epochs=args.epochs,
            device=training_device,
        )
        imitation_loss = pretraining_result.behavior_loss
        report["stages"]["behavior_cloning"] = {
            "examples": len(examples),
            "loss": imitation_loss,
            "decision_type_counts": dict(pretraining_result.decision_type_counts),
            "elapsed_seconds": round(perf_counter() - stage_started, 2),
        }
        write_json(args.report, report)
        print(
            f"[Stage 1/4 complete] examples={len(examples)} "
            f"loss={imitation_loss:.6f} "
            f"elapsed={(perf_counter() - stage_started) / 60:.1f}m",
            flush=True,
        )

        replay_buffer = DecisionReplayBuffer(
            [
                DecisionReplayRecord(
                    season=args.start_season,
                    week=0,
                    decision_type=example.decision_type,
                    action_key=str(index),
                    features=example.features,
                    reward=example.target_score,
                    source="historical_teacher",
                )
                for index, example in enumerate(examples)
            ]
        )
        stage_started = perf_counter()
        print("[Stage 2/4] Collecting historical replay...", flush=True)
        if args.collect_season_replay:
            replay_buffer.extend(collect_season_replay(seasons, teacher.genome).records)
        report["stages"]["replay_collection"] = {
            "records": len(replay_buffer),
            "elapsed_seconds": round(perf_counter() - stage_started, 2),
        }
        write_json(args.report, report)
        print(
            f"[Stage 2/4 complete] records={len(replay_buffer)} "
            f"elapsed={(perf_counter() - stage_started) / 60:.1f}m",
            flush=True,
        )

        transaction_value_model = None
        transaction_value_model_approved = False
        transaction_records = [
            record
            for record in replay_buffer.records
            if record.decision_type in ("waiver", "trade")
        ]
        if transaction_records:
            stage_started = perf_counter()
            print("[Stage 2.5/4] Training transaction value-risk model...", flush=True)
            transaction_value_model = TransactionValueNetwork()
            (
                transaction_value_loss,
                transaction_record_count,
                transaction_value_validation,
            ) = train_transaction_value_model_with_validation(
                model=transaction_value_model,
                records=transaction_records,
                epochs=args.transaction_value_epochs,
                device=training_device,
                holdout_seasons=args.transaction_value_validation_seasons,
                minimum_validation_records=args.transaction_value_min_validation_records,
            )
            transaction_value_path = save_transaction_value_model(
                transaction_value_model,
                args.transaction_value_output,
            )
            report["stages"]["transaction_value_training"] = {
                "records": transaction_record_count,
                "executed_records": sum(record.executed for record in transaction_records),
                "rejected_or_unexecuted_records": sum(
                    not record.executed for record in transaction_records
                ),
                "loss": transaction_value_loss,
                "elapsed_seconds": round(perf_counter() - stage_started, 2),
                "model_path": str(transaction_value_path),
                "validation": transaction_value_validation.to_dict(),
            }
            transaction_value_model_approved = transaction_value_validation.approved
            write_json(args.report, report)
            print(
                f"[Stage 2.5/4 complete] records={transaction_record_count} "
                f"loss={transaction_value_loss:.6f} "
                f"validation_approved={transaction_value_model_approved} "
                f"elapsed={(perf_counter() - stage_started) / 60:.1f}m",
                flush=True,
            )
        else:
            report["stages"]["transaction_value_training"] = {
                "records": 0,
                "skipped": True,
                "reason": "No waiver/trade replay records were collected.",
            }
            write_json(args.report, report)

        requested_transaction_mode = args.transaction_mode
        effective_transaction_mode = requested_transaction_mode
        if requested_transaction_mode == "hybrid" and not transaction_value_model_approved:
            effective_transaction_mode = "genome"
            print(
                "[Safety gate] Transaction value model failed chronological validation; "
                "using genome transactions for self-play and final evaluation.",
                flush=True,
            )
        report["configuration"]["effective_transaction_mode"] = effective_transaction_mode
        report["configuration"]["transaction_value_model_approved"] = (
            transaction_value_model_approved
        )
        write_json(args.report, report)

        stage_started = perf_counter()
        print("[Stage 3/4] Offline replay training...", flush=True)
        offline_loss = train_offline_policy(
            model, replay_buffer, epochs=args.offline_epochs, device=training_device
        )
        report["stages"]["offline_training"] = {
            "loss": offline_loss,
            "elapsed_seconds": round(perf_counter() - stage_started, 2),
        }
        write_json(args.report, report)
        print(
            f"[Stage 3/4 complete] loss={offline_loss:.6f} "
            f"elapsed={(perf_counter() - stage_started) / 60:.1f}m",
            flush=True,
        )

        stage_started = perf_counter()
        print("[Stage 4/4] Self-play evolution starting...", flush=True)
        # The simulator is intentionally CPU/process based.  Moving the small
        # policy to CPU before it crosses process boundaries avoids CUDA context
        # duplication and lets CUDA be reserved for the batched training stages.
        model.to("cpu")
        if transaction_value_model is not None:
            transaction_value_model.to("cpu")
        scenarios = [
            (
                create_training_league(season),
                load_weekly_performances(season, include_special_teams=True),
            )
            for season in seasons
        ]
        trained_model, history = train_modular_policy_self_play(
            initial_policy=model,
            scenarios=scenarios,
            transaction_genome=teacher.genome,
            population_size=args.population,
            generations=args.generations,
            selection_count=args.selection,
            mutation_strength=args.mutation_strength,
            final_mutation_strength=args.final_mutation_strength,
            risk_penalty=args.risk_penalty,
            elite_count=args.elite_count,
            draft_exploration_rate=args.draft_exploration_rate,
            draft_exploration_top_k=args.draft_exploration_top_k,
                    diversity_floor=args.diversity_floor,
                    diversity_mutation_boost=args.diversity_mutation_boost,
                    baseline_relative_weight=args.baseline_relative_weight,
                    immigrant_fraction=args.immigrant_fraction,
            transaction_ablation=args.transaction_ablation,
            transaction_mode=effective_transaction_mode,
            transaction_value_model=(
                transaction_value_model if transaction_value_model_approved else None
            ),
            seed=args.seed,
            rounds=args.rounds,
            evaluation_workers=args.evaluation_workers,
            lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
            scenarios_per_generation=(
                args.scenarios_per_generation if args.scenarios_per_generation > 0 else None
            ),
            full_evaluation_interval=args.full_evaluation_every,
            anchor_scenarios_per_generation=args.anchor_scenarios_per_generation,
            final_selection_count=args.final_selection_count,
            candidate_archive_size=args.candidate_archive_size,
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
                metadata={
                    "start_season": args.start_season,
                    "end_season": args.end_season,
                    "population": args.population,
                    "selection": args.selection,
                    "seed": args.seed,
                    "mutation_strength": args.mutation_strength,
                    "final_mutation_strength": args.final_mutation_strength,
                    "risk_penalty": args.risk_penalty,
                    "elite_count": args.elite_count,
                    "draft_exploration_rate": args.draft_exploration_rate,
                    "draft_exploration_top_k": args.draft_exploration_top_k,
                    "diversity_floor": args.diversity_floor,
                    "diversity_mutation_boost": args.diversity_mutation_boost,
                    "baseline_relative_weight": args.baseline_relative_weight,
                    "immigrant_fraction": args.immigrant_fraction,
                    "transaction_ablation": args.transaction_ablation,
                    "transaction_mode": effective_transaction_mode,
                    "transaction_value_output": str(args.transaction_value_output),
                    "rounds": args.rounds,
                    "evaluation_workers": args.evaluation_workers,
                    "scenarios_per_generation": args.scenarios_per_generation,
                    "full_evaluation_every": args.full_evaluation_every,
                    "anchor_scenarios_per_generation": args.anchor_scenarios_per_generation,
                    "final_selection_count": args.final_selection_count,
                    "candidate_archive_size": args.candidate_archive_size,
                },
            ),
        )
        report["stages"]["self_play"] = {
            "history": history,
            "elapsed_seconds": round(perf_counter() - stage_started, 2),
        }
        model_path = save_modular_policy_network(trained_model, args.output)
        report["status"] = "completed"
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report["elapsed_seconds"] = round(perf_counter() - run_started, 2)
        report["model_path"] = str(model_path)
        report["updated_at"] = report["finished_at"]
        write_json(args.report, report)
        try:
            print(
                f"[Stage 4/4 complete] elapsed={(perf_counter() - stage_started) / 3600:.2f}h",
                flush=True,
            )
            print("Modular manager policy training complete", flush=True)
            print(f"Policy saved to: {model_path}", flush=True)
            print(f"Structured report saved to: {args.report}", flush=True)
        except OSError:
            # The report/checkpoint are already durable.  A closed terminal or
            # Tee pipeline must not turn a completed overnight run into a
            # reported failure.
            pass
    except Exception as error:
        report["status"] = "failed"
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report["elapsed_seconds"] = round(perf_counter() - run_started, 2)
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        report["updated_at"] = report["finished_at"]
        write_json(args.report, report)
        raise


if __name__ == "__main__":
    main()
