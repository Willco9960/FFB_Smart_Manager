from pathlib import Path

from evolution.genome import DraftStrategyGenome, create_random_genome
from evolution.multi_season_evaluation import evaluate_neural_manager_for_season
from evolution.projection_mode_comparison import (
    ProjectionComparison,
    ProjectionModeComparison,
    format_projection_comparison,
)
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.lineup import ESPN_DEFAULT_LINEUP_RULES
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import load_weekly_performances
from models.draft_projection_nn import DEFAULT_MODEL_PATH
from models.manager_policy_nn import load_manager_policy_network
from models.projection_service import load_neural_projection_service
from models.weekly_projection_service import load_weekly_projection_service

POLICY_MODEL_PATH = Path("data/models/manager_policy_full_season.pt")
FALLBACK_POLICY_PATH = Path("data/models/manager_policy_network.pt")
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
        name=f"{season} Projection Comparison League",
        teams=[Team(name=f"Comparison Team {number}") for number in range(1, 11)],
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


def main():
    policy_path = POLICY_MODEL_PATH if POLICY_MODEL_PATH.exists() else FALLBACK_POLICY_PATH
    policy_network = load_manager_policy_network(policy_path)
    transaction_genome = load_transaction_genome()
    comparisons = []

    for season in EVALUATION_SEASONS:
        print(f"Comparing projection modes for {season}...")
        performances = load_weekly_performances(season, include_special_teams=True)
        weekly_service = load_weekly_projection_service(
            model_path=WEEKLY_MODEL_PATH,
            target_season=season,
        )

        if weekly_service is None:
            raise FileNotFoundError(
                f"Weekly projection checkpoint not found: {WEEKLY_MODEL_PATH}"
            )

        heuristic_result = evaluate_neural_manager_for_season(
            season=season,
            policy_network=policy_network,
            transaction_genome=transaction_genome,
            league=create_season_league(season),
            performances=performances,
            lineup_rules=ESPN_DEFAULT_LINEUP_RULES,
            seed=season,
        )
        weekly_neural_result = evaluate_neural_manager_for_season(
            season=season,
            policy_network=policy_network,
            transaction_genome=transaction_genome,
            league=create_season_league(season),
            performances=performances,
            lineup_rules=ESPN_DEFAULT_LINEUP_RULES,
            seed=season,
            projection_service=weekly_service,
        )
        comparisons.append(
            ProjectionModeComparison(
                season=season,
                heuristic=heuristic_result,
                weekly_neural=weekly_neural_result,
            )
        )

    print(format_projection_comparison(ProjectionComparison(tuple(comparisons))))


if __name__ == "__main__":
    main()
