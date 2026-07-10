from pathlib import Path

from evolution.training import (
    format_generation_roster_report,
    format_training_log,
    run_and_save_training_experiment,
)
from evolution.training_graph import save_training_progress_graph
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES
from fantasy_engine.team import Team

OUTPUT_PATH = Path("data/evolution/best_2021_real_data_genome.json")
GRAPH_OUTPUT_PATH = Path("reports/2021_real_training_progress.png")


def create_2021_training_league() -> League:
    teams = [Team(name=f"Real Data Team {number}") for number in range(1, 11)]
    players = load_leakage_safe_player_pool(
        projection_season=2020,
        actual_season=2021,
    )[:250]

    return League(
        name="2021 Leakage-Safe Training League",
        teams=teams,
        available_players=players,
    )


def main():
    generation_count = 5

    league = create_2021_training_league()
    training_result = run_and_save_training_experiment(
        league=league,
        output_path=OUTPUT_PATH,
        population_size=20,
        generation_count=generation_count,
        selection_count=5,
        mutation_strength=0.12,
        final_mutation_strength=0.03,
        seed=2021,
        rounds=16,
        lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
    )

    print(format_training_log(training_result))
    print("Using temporary offensive lineup rules: QB, 2 RB, 2 WR, TE, FLEX")
    print("K and DST will be added after historical K/DST scoring is added.")
    print("")

    last_generation = len(training_result.generation_results)
    first_report_generation = max(1, last_generation - 2)
    report_generations = list(range(first_report_generation, last_generation + 1))

    print(f"Generation {report_generations[0]}-{report_generations[-1]} roster comparison")

    print(format_generation_roster_report(training_result, report_generations))
    graph_path = save_training_progress_graph(
        training_result=training_result,
        output_path=GRAPH_OUTPUT_PATH,
    )
    print(f"Training graph saved to: {graph_path}")
    print(f"Best genome saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
