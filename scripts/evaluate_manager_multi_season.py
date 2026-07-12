from pathlib import Path

from evolution.genome import DraftStrategyGenome, create_random_genome
from evolution.multi_season_evaluation import (
    MultiSeasonEvaluation,
    evaluate_neural_manager_for_season,
    format_multi_season_evaluation,
)
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.lineup import ESPN_DEFAULT_LINEUP_RULES
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import load_weekly_performances
from models.draft_projection_nn import DEFAULT_MODEL_PATH
from models.manager_policy_nn import load_manager_policy_network
from models.projection_service import load_neural_projection_service

POLICY_MODEL_PATH = Path("data/models/manager_policy_full_season.pt")
FALLBACK_POLICY_PATH = Path("data/models/manager_policy_network.pt")
TRANSACTION_GENOME_PATH = Path("data/evolution/best_full_season_2021_genome.json")
EVALUATION_SEASONS = (2021, 2022, 2023, 2024, 2025)


def create_season_league(season: int) -> League:
    players = load_leakage_safe_player_pool(
        projection_season=season - 1,
        actual_season=season,
        include_special_teams=True,
    )[:250]
    league = League(
        name=f"{season} Evaluation League",
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


def main():
    policy_path = POLICY_MODEL_PATH if POLICY_MODEL_PATH.exists() else FALLBACK_POLICY_PATH
    policy_network = load_manager_policy_network(policy_path)
    transaction_genome = load_transaction_genome()
    results = []

    for season in EVALUATION_SEASONS:
        print(f"Evaluating season {season}...")
        results.append(
            evaluate_neural_manager_for_season(
                season=season,
                policy_network=policy_network,
                transaction_genome=transaction_genome,
                league=create_season_league(season),
                performances=load_weekly_performances(
                    season,
                    include_special_teams=True,
                ),
                lineup_rules=ESPN_DEFAULT_LINEUP_RULES,
                seed=season,
            )
        )

    print(format_multi_season_evaluation(MultiSeasonEvaluation(tuple(results))))


if __name__ == "__main__":
    main()
