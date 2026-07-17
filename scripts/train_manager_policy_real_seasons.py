import argparse
from pathlib import Path

from evolution.genome import DraftStrategyGenome, create_random_genome
from evolution.neural_policy_training import (
    DEFAULT_CONSISTENCY_PENALTY,
    NeuralGenerationResult,
    NeuralPolicySeasonScenario,
    NeuralTrainingProgress,
    train_neural_policy_across_seasons,
)
from fantasy_engine.historical_seasons import (
    EARLIEST_RELIABLE_SEASON,
    get_training_seasons,
)
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.lineup import ESPN_DEFAULT_LINEUP_RULES
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import load_weekly_performances
from models.draft_projection_nn import DEFAULT_MODEL_PATH
from models.manager_policy_nn import load_manager_policy_network, save_manager_policy_network
from models.projection_service import load_neural_projection_service
from models.weekly_projection_service import load_weekly_projection_service

INITIAL_POLICY_PATHS = (
    Path("data/models/manager_policy_full_season.pt"),
    Path("data/models/manager_policy_network.pt"),
)
OUTPUT_PATH = Path("data/models/manager_policy_real_seasons.pt")
CHECKPOINT_DIR = Path("data/models/manager_training_checkpoints")
TRAINING_START_SEASON = EARLIEST_RELIABLE_SEASON
TRAINING_END_SEASON = 2024
TRAINING_SEASONS = get_training_seasons(TRAINING_START_SEASON, TRAINING_END_SEASON)
HOLDOUT_SEASON = 2025
WEEKLY_MODEL_PATH = Path("data/models/weekly_projection_network.pt")
DRAFT_MODEL_MIN_TARGET_SEASON = 2020
WEEKLY_MODEL_MIN_TARGET_SEASON = 2020


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a manager on real historical seasons.")
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--selection", type=int, default=3)
    parser.add_argument("--mutation", type=float, default=0.02)
    parser.add_argument("--consistency-penalty", type=float, default=DEFAULT_CONSISTENCY_PENALTY)
    parser.add_argument("--start-season", type=int, default=TRAINING_START_SEASON)
    parser.add_argument("--end-season", type=int, default=TRAINING_END_SEASON)
    parser.add_argument("--holdout-season", type=int, default=HOLDOUT_SEASON)
    parser.add_argument(
        "--initial-policy",
        type=Path,
        default=None,
        help="Optional policy checkpoint to continue training from.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
        help="Directory for completed-generation policy checkpoints.",
    )
    return parser.parse_args()


def load_initial_policy(requested_path: Path | None = None):
    if requested_path is not None:
        if not requested_path.exists():
            raise FileNotFoundError(f"Initial policy checkpoint not found: {requested_path}")
        return requested_path, load_manager_policy_network(requested_path)

    for path in INITIAL_POLICY_PATHS:
        if path.exists():
            return path, load_manager_policy_network(path)

    raise FileNotFoundError("No initial manager policy checkpoint was found.")


def load_transaction_genome() -> DraftStrategyGenome:
    path = Path("data/evolution/best_full_season_2021_genome.json")

    if path.exists():
        return DraftStrategyGenome.from_json(path.read_text(encoding="utf-8"))

    return create_random_genome(seed=2021)


def create_generation_checkpoint_callback(checkpoint_dir: Path):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_generation_checkpoint(generation: NeuralGenerationResult) -> None:
        generation_path = checkpoint_dir / f"generation_{generation.generation_number:03d}.pt"
        latest_path = checkpoint_dir / "latest.pt"
        save_manager_policy_network(generation.best_agent.policy_network, generation_path)
        save_manager_policy_network(generation.best_agent.policy_network, latest_path)

    return save_generation_checkpoint


def create_season_scenario(season: int) -> NeuralPolicySeasonScenario:
    league = League(
        name=f"{season} Real Training League",
        teams=[Team(name=f"Real Training Team {number}") for number in range(1, 11)],
        available_players=load_leakage_safe_player_pool(
            projection_season=season - 1,
            actual_season=season,
            include_special_teams=True,
        )[:250],
    )
    draft_projection_service = None

    if season >= DRAFT_MODEL_MIN_TARGET_SEASON:
        draft_projection_service = load_neural_projection_service(
            DEFAULT_MODEL_PATH,
            target_season=season,
        )

    weekly_projection_service = None

    if season >= WEEKLY_MODEL_MIN_TARGET_SEASON:
        weekly_projection_service = load_weekly_projection_service(
            model_path=WEEKLY_MODEL_PATH,
            target_season=season,
        )

    if draft_projection_service is not None:
        league = draft_projection_service.project_league(league)

    return NeuralPolicySeasonScenario(
        season=season,
        league=league,
        performances=load_weekly_performances(season, include_special_teams=True),
        projection_service=weekly_projection_service,
    )


def format_elapsed(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_training_progress(progress: NeuralTrainingProgress) -> None:
    elapsed = format_elapsed(progress.elapsed_seconds)

    if progress.status == "generation_complete":
        print(
            f"\nGeneration {progress.generation_number}/{progress.generation_count} complete | "
            f"average {progress.average_fitness:.2f} | "
            f"best {progress.best_fitness:.2f} | elapsed {elapsed}",
            flush=True,
        )
        print(
            f"  avg wins {progress.average_wins:.2f} | "
            f"avg PF {progress.average_points_for:.2f} | "
            f"playoffs {progress.playoff_rate:.1%} | "
            f"championship rate {progress.championship_rate:.1%} | "
            f"best risk-adjusted {progress.best_risk_adjusted_fitness:.2f}",
            flush=True,
        )
        print(
            f"  best agent: wins {progress.best_agent_average_wins:.2f} | "
            f"PF {progress.best_agent_average_points_for:.2f} | "
            f"playoffs {progress.best_agent_playoff_rate:.1%} | "
            f"championships {progress.best_agent_championship_rate:.1%} | "
            f"fitness stddev {progress.average_fitness_stddev:.2f}",
            flush=True,
        )
        return

    marker = "Starting" if progress.status == "starting" else "Completed"
    print(
        f"[{marker}] generation {progress.generation_number}/{progress.generation_count} | "
        f"scenario {progress.scenario_number}/{progress.scenario_count} | "
        f"elapsed {elapsed}",
        flush=True,
    )


def main():
    args = parse_args()
    training_seasons = get_training_seasons(args.start_season, args.end_season)

    if args.holdout_season <= max(training_seasons):
        raise ValueError("Holdout season must be after every training season.")

    initial_policy_path, initial_network = load_initial_policy(args.initial_policy)
    scenarios = [create_season_scenario(season) for season in training_seasons]
    training_result = train_neural_policy_across_seasons(
        initial_network=initial_network,
        scenarios=scenarios,
        transaction_genome=load_transaction_genome(),
        population_size=args.population,
        generation_count=args.generations,
        selection_count=args.selection,
        mutation_strength=args.mutation,
        consistency_penalty=args.consistency_penalty,
        seed=2021,
        rounds=16,
        lineup_rules=ESPN_DEFAULT_LINEUP_RULES,
        progress_callback=print_training_progress,
        generation_callback=create_generation_checkpoint_callback(args.checkpoint_dir),
    )
    save_manager_policy_network(training_result.best_agent.policy_network, OUTPUT_PATH)

    print("Real-season neural manager training complete")
    print(f"Initial policy: {initial_policy_path}")
    print(f"Training seasons: {training_seasons}")
    print(f"Holdout season: {args.holdout_season}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Selected policy generation: {training_result.best_generation_number}")
    print(f"Generation checkpoints: {args.checkpoint_dir}")

    for generation in training_result.generations:
        print(
            f"Generation {generation.generation_number}: "
            f"average fitness = {generation.average_fitness:.2f}, "
            f"best fitness = {generation.best_fitness:.2f}, "
            f"best risk-adjusted = {generation.best_risk_adjusted_fitness:.2f}, "
            f"average wins = {generation.average_wins:.2f}, "
            f"playoff rate = {generation.playoff_rate:.1%}, "
            f"championship rate = {generation.championship_rate:.1%}"
        )
        print(
            f"  best agent: wins {generation.best_agent_average_wins:.2f}, "
            f"PF {generation.best_agent_average_points_for:.2f}, "
            f"playoffs {generation.best_agent_playoff_rate:.1%}, "
            f"championships {generation.best_agent_championship_rate:.1%} | "
            f"population fitness stddev {generation.average_fitness_stddev:.2f}"
        )

    print(f"Policy model saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
