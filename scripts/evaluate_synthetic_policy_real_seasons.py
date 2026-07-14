from pathlib import Path

from evolution.genome import DraftStrategyGenome, create_random_genome
from evolution.multi_season_evaluation import evaluate_neural_manager_for_season
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.lineup import ESPN_DEFAULT_LINEUP_RULES
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import load_weekly_performances
from models.draft_projection_nn import DEFAULT_MODEL_PATH
from models.manager_policy_nn import load_manager_policy_network
from models.projection_service import load_neural_projection_service
from models.weekly_projection_service import load_weekly_projection_service

ORIGINAL_POLICY_PATH = Path("data/models/manager_policy_full_season.pt")
SYNTHETIC_POLICY_PATH = Path("data/models/manager_policy_synthetic_seasons.pt")
TRANSACTION_GENOME_PATH = Path("data/evolution/best_full_season_2021_genome.json")
WEEKLY_MODEL_PATH = Path("data/models/weekly_projection_network.pt")
EVALUATION_SEASONS = (2021, 2022, 2023, 2024, 2025)


def create_season_league(season: int) -> League:
    players = load_leakage_safe_player_pool(
        projection_season=season - 1,
        actual_season=season,
        include_special_teams=True,
    )[:250]
    league = League(
        name=f"{season} Synthetic Policy Evaluation League",
        teams=[Team(name=f"Evaluation Team {number}") for number in range(1, 11)],
        available_players=players,
    )
    projection_service = load_neural_projection_service(DEFAULT_MODEL_PATH)

    if projection_service is not None:
        return projection_service.project_league(league)

    return league


def load_transaction_genome() -> DraftStrategyGenome:
    if TRANSACTION_GENOME_PATH.exists():
        return DraftStrategyGenome.from_json(
            TRANSACTION_GENOME_PATH.read_text(encoding="utf-8")
        )

    return create_random_genome(seed=2021)


def evaluate_policy(
    policy_path: Path,
    season: int,
    transaction_genome: DraftStrategyGenome,
):
    performances = load_weekly_performances(season, include_special_teams=True)
    projection_service = load_weekly_projection_service(
        model_path=WEEKLY_MODEL_PATH,
        target_season=season,
    )

    return evaluate_neural_manager_for_season(
        season=season,
        policy_network=load_manager_policy_network(policy_path),
        transaction_genome=transaction_genome,
        league=create_season_league(season),
        performances=performances,
        lineup_rules=ESPN_DEFAULT_LINEUP_RULES,
        seed=season,
        projection_service=projection_service,
    )


def main():
    if not ORIGINAL_POLICY_PATH.exists():
        raise FileNotFoundError(f"Original policy not found: {ORIGINAL_POLICY_PATH}")
    if not SYNTHETIC_POLICY_PATH.exists():
        raise FileNotFoundError(f"Synthetic policy not found: {SYNTHETIC_POLICY_PATH}")

    transaction_genome = load_transaction_genome()
    original_results = []
    synthetic_results = []

    for season in EVALUATION_SEASONS:
        print(f"Evaluating both policies on real season {season}...")
        original_results.append(
            evaluate_policy(ORIGINAL_POLICY_PATH, season, transaction_genome)
        )
        synthetic_results.append(
            evaluate_policy(SYNTHETIC_POLICY_PATH, season, transaction_genome)
        )

    print("\nSynthetic-trained policy versus original policy")

    for original, synthetic in zip(original_results, synthetic_results, strict=True):
        print(
            f"{original.season}: "
            f"original {original.fitness:.2f}, "
            f"synthetic-trained {synthetic.fitness:.2f}, "
            f"delta {synthetic.fitness - original.fitness:+.2f}, "
            f"wins {original.wins}->{synthetic.wins}, "
            f"champion {original.champion}->{synthetic.champion}"
        )

    original_average = sum(result.fitness for result in original_results) / len(original_results)
    synthetic_average = sum(result.fitness for result in synthetic_results) / len(
        synthetic_results
    )
    original_playoffs = sum(result.playoff_seed is not None for result in original_results)
    synthetic_playoffs = sum(result.playoff_seed is not None for result in synthetic_results)
    original_championships = sum(result.champion for result in original_results)
    synthetic_championships = sum(result.champion for result in synthetic_results)

    print(f"Average fitness: {original_average:.2f}->{synthetic_average:.2f}")
    print(f"Playoff appearances: {original_playoffs}/5->{synthetic_playoffs}/5")
    print(f"Championships: {original_championships}/5->{synthetic_championships}/5")


if __name__ == "__main__":
    main()
