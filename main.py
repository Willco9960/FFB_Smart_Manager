from fantasy_engine.draft import format_draft_results, run_snake_draft
from fantasy_engine.fake_data import create_fake_player_pool
from fantasy_engine.league import League
from fantasy_engine.team import Team

APP_NAME = "Fantasy Football AI Manager"
APP_VERSION = "0.1.0"
CURRENT_PHASE = "Phase 1 - Basic Fantasy Simulator"


def main():
    players = create_fake_player_pool()
    teams = [Team(name=f"Team {number}") for number in range(1, 11)]

    league = League(
        name="Demo League",
        teams=teams,
        available_players=players,
    )

    draft_results = run_snake_draft(league, rounds=16)

    print(f"{APP_NAME} v{APP_VERSION}")
    print(CURRENT_PHASE)
    print("Status: ready for local development")
    print(f"Fake player pool loaded: {len(players) + len(draft_results)} players")
    print(f"Draft completed: {len(draft_results)} picks")
    print()
    print(format_draft_results(draft_results[:10]))


if __name__ == "__main__":
    main()
