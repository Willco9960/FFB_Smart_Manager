from evolution.training import (
    DEFAULT_OUTPUT_PATH,
    format_training_log,
    run_and_save_training_experiment,
)
from fantasy_engine.fake_data import create_fake_player_pool
from fantasy_engine.league import League
from fantasy_engine.team import Team


def create_training_league() -> League:
    teams = [Team(name=f"Training Team {number}") for number in range(1, 11)]

    return League(
        name="Training League",
        teams=teams,
        available_players=create_fake_player_pool(),
    )


def main():
    league = create_training_league()
    training_result = run_and_save_training_experiment(league)

    print(format_training_log(training_result))
    print(f"Best genome saved to: {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
