from pathlib import Path

from evolution.genome import DraftStrategyGenome, create_random_genome
from evolution.neural_policy_training import (
    NeuralPolicySeasonScenario,
    train_neural_policy_across_seasons,
)
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.lineup import ESPN_DEFAULT_LINEUP_RULES
from fantasy_engine.synthetic_season import generate_synthetic_seasons
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import load_weekly_performances
from models.draft_projection_nn import DEFAULT_MODEL_PATH
from models.manager_policy_nn import load_manager_policy_network, save_manager_policy_network
from models.projection_service import load_neural_projection_service
from models.weekly_projection_service import load_weekly_projection_service

INITIAL_POLICY_PATH = Path("data/models/manager_policy_network.pt")
OUTPUT_PATH = Path("data/models/manager_policy_mixed_seasons.pt")
TRANSACTION_GENOME_PATH = Path("data/evolution/best_full_season_2021_genome.json")
WEEKLY_MODEL_PATH = Path("data/models/weekly_projection_network.pt")
TRAINING_SEASONS = (2021, 2022, 2023, 2024)
SYNTHETIC_SOURCE_SEASON = 2021


def load_transaction_genome() -> DraftStrategyGenome:
    if TRANSACTION_GENOME_PATH.exists():
        return DraftStrategyGenome.from_json(
            TRANSACTION_GENOME_PATH.read_text(encoding="utf-8")
        )

    return create_random_genome(seed=2021)


def create_season_scenario(season: int) -> NeuralPolicySeasonScenario:
    league = League(
        name=f"{season} Mixed Training League",
        teams=[Team(name=f"Mixed Team {number}") for number in range(1, 11)],
        available_players=load_leakage_safe_player_pool(
            projection_season=season - 1,
            actual_season=season,
            include_special_teams=True,
        )[:250],
    )
    draft_projection_service = load_neural_projection_service(DEFAULT_MODEL_PATH)

    if draft_projection_service is not None:
        league = draft_projection_service.project_league(league)

    return NeuralPolicySeasonScenario(
        season=season,
        league=league,
        performances=load_weekly_performances(
            season,
            include_special_teams=True,
        ),
        projection_service=load_weekly_projection_service(
            model_path=WEEKLY_MODEL_PATH,
            target_season=season,
        ),
    )


def main():
    scenarios = [create_season_scenario(season) for season in TRAINING_SEASONS]
    synthetic_source = next(
        scenario for scenario in scenarios if scenario.season == SYNTHETIC_SOURCE_SEASON
    )
    synthetic_performances = generate_synthetic_seasons(
        performances=synthetic_source.performances,
        season_count=1,
        seed=SYNTHETIC_SOURCE_SEASON,
    )
    synthetic_scenario = NeuralPolicySeasonScenario(
        season=synthetic_source.season,
        league=synthetic_source.league,
        performances=synthetic_source.performances,
        synthetic_performances=synthetic_performances,
        projection_service=synthetic_source.projection_service,
    )
    scenarios = [
        scenario
        for scenario in scenarios
        if scenario.season != SYNTHETIC_SOURCE_SEASON
    ]
    scenarios.append(synthetic_scenario)

    training_result = train_neural_policy_across_seasons(
        initial_network=load_manager_policy_network(INITIAL_POLICY_PATH),
        scenarios=scenarios,
        transaction_genome=load_transaction_genome(),
        population_size=10,
        generation_count=2,
        selection_count=3,
        mutation_strength=0.02,
        seed=2021,
        rounds=16,
        lineup_rules=ESPN_DEFAULT_LINEUP_RULES,
    )
    save_manager_policy_network(training_result.best_agent.policy_network, OUTPUT_PATH)

    print("Mixed real-plus-synthetic neural manager training complete")
    print(f"Real seasons: {TRAINING_SEASONS}")
    print("Synthetic share: 1 of 5 scenarios (20%)")

    for generation in training_result.generations:
        print(
            f"Generation {generation.generation_number}: "
            f"average scenario fitness = {generation.average_fitness:.2f}, "
            f"best scenario fitness = {generation.best_fitness:.2f}"
        )

    print(f"Policy model saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
