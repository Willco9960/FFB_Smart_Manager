from pathlib import Path

from evolution.training import format_training_log, run_and_save_training_experiment
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import load_weekly_performances

OUTPUT_PATH = Path("data/evolution/best_full_season_2021_genome.json")


def create_training_league() -> League:
    teams = [Team(name=f"Full Season Team {number}") for number in range(1, 11)]
    players = load_leakage_safe_player_pool(2020, 2021)[:250]

    return League(
        name="2021 Full-Season Training League",
        teams=teams,
        available_players=players,
    )


def main():
    training_result = run_and_save_training_experiment(
        league=create_training_league(),
        output_path=OUTPUT_PATH,
        population_size=10,
        generation_count=2,
        selection_count=3,
        mutation_strength=0.1,
        final_mutation_strength=0.03,
        seed=2021,
        rounds=16,
        lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
        performances=load_weekly_performances(2021),
    )

    print("Full-season evolutionary training complete")
    print(format_training_log(training_result))
    print(f"Best genome saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
