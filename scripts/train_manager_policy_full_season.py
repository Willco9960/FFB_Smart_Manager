from pathlib import Path

from evolution.genome import DraftStrategyGenome, create_random_genome
from evolution.neural_policy_training import train_neural_policy_on_seasons
from fantasy_engine.lineup import ESPN_DEFAULT_LINEUP_RULES
from fantasy_engine.weekly_data import load_weekly_performances
from models.manager_policy_nn import load_manager_policy_network, save_manager_policy_network
from scripts.run_full_season_training_experiment import create_training_league

POLICY_MODEL_PATH = Path("data/models/manager_policy_network.pt")
OUTPUT_PATH = Path("data/models/manager_policy_full_season.pt")
TRANSACTION_GENOME_PATH = Path("data/evolution/best_full_season_2021_genome.json")


def load_transaction_genome() -> DraftStrategyGenome:
    if TRANSACTION_GENOME_PATH.exists():
        return DraftStrategyGenome.from_json(
            TRANSACTION_GENOME_PATH.read_text(encoding="utf-8")
        )

    return create_random_genome(seed=2021)


def main():
    initial_network = load_manager_policy_network(POLICY_MODEL_PATH)
    training_result = train_neural_policy_on_seasons(
        initial_network=initial_network,
        league=create_training_league(),
        performances=load_weekly_performances(2021, include_special_teams=True),
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

    print("Reward-based neural manager training complete")

    for generation in training_result.generations:
        print(
            f"Generation {generation.generation_number}: "
            f"best fitness = {generation.best_fitness:.2f}, "
            f"average fitness = {generation.average_fitness:.2f}"
        )

    print(f"Policy model saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
