from pathlib import Path

from evolution.neural_policy_training import train_neural_policy_on_seasons
from fantasy_engine.lineup import ESPN_DEFAULT_LINEUP_RULES
from fantasy_engine.synthetic_season import generate_synthetic_seasons
from fantasy_engine.weekly_data import load_weekly_performances
from models.manager_policy_nn import load_manager_policy_network, save_manager_policy_network
from scripts.run_full_season_training_experiment import create_training_league
from scripts.train_manager_policy_full_season import load_transaction_genome

INITIAL_POLICY_PATH = Path("data/models/manager_policy_network.pt")
OUTPUT_PATH = Path("data/models/manager_policy_synthetic_seasons.pt")
REAL_SEASON = 2021
SYNTHETIC_SEASON_COUNT = 3


def main():
    real_performances = load_weekly_performances(
        REAL_SEASON,
        include_special_teams=True,
    )
    synthetic_performances = generate_synthetic_seasons(
        performances=real_performances,
        season_count=SYNTHETIC_SEASON_COUNT,
        seed=2021,
    )
    training_result = train_neural_policy_on_seasons(
        initial_network=load_manager_policy_network(INITIAL_POLICY_PATH),
        league=create_training_league(),
        performances=real_performances,
        synthetic_performances=synthetic_performances,
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

    print("Synthetic-augmented neural manager training complete")
    print(f"Real seasons: 1 ({REAL_SEASON})")
    print(f"Synthetic seasons per generation: {SYNTHETIC_SEASON_COUNT}")

    for generation in training_result.generations:
        print(
            f"Generation {generation.generation_number}: "
            f"average scenario fitness = {generation.average_fitness:.2f}, "
            f"best scenario fitness = {generation.best_fitness:.2f}"
        )

    print(f"Policy model saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
