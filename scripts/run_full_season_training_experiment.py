from pathlib import Path

from evolution.training import format_training_log, run_and_save_training_experiment
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.lineup import ESPN_DEFAULT_LINEUP_RULES
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import load_weekly_performances
from models.draft_projection_nn import DEFAULT_MODEL_PATH
from models.projection_service import load_neural_projection_service

OUTPUT_PATH = Path("data/evolution/best_full_season_2021_genome.json")


def create_training_league() -> League:
    teams = [Team(name=f"Full Season Team {number}") for number in range(1, 11)]
    league = League(
        name="2021 Full-Season Training League",
        teams=teams,
        available_players=load_leakage_safe_player_pool(
            2020,
            2021,
            include_special_teams=True,
        )[:250],
    )

    projection_service = load_neural_projection_service(DEFAULT_MODEL_PATH)

    if projection_service is None:
        print("Projection source: leakage-safe historical projections")
        return league

    print("Projection source: neural projection network")
    return projection_service.project_league(league)


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
        lineup_rules=ESPN_DEFAULT_LINEUP_RULES,
        performances=load_weekly_performances(2021, include_special_teams=True),
    )

    print("Full-season evolutionary training complete")
    print(format_training_log(training_result))
    print(f"Best genome saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
