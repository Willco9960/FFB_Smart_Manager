from agents.random_agent import RandomDraftAgent
from fantasy_engine.draft import format_draft_results, run_snake_draft
from fantasy_engine.fake_data import create_fake_player_pool
from fantasy_engine.league import League
from fantasy_engine.scoring import format_ranked_team_scores, get_winner, score_teams
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

    draft_agent = RandomDraftAgent(seed=42)
    draft_results = run_snake_draft(league, rounds=16, draft_agent=draft_agent)
    team_scores = score_teams(league.teams)
    winner = get_winner(team_scores)

    print(f"{APP_NAME} v{APP_VERSION}")
    print(CURRENT_PHASE)
    print("Status: ready for local development")
    print(f"Fake player pool loaded: {len(players) + len(draft_results)} players")
    print(f"Draft completed: {len(draft_results)} picks")
    print("Draft agent: RandomDraftAgent")
    print()
    print("First 10 draft picks:")
    print(format_draft_results(draft_results[:10]))
    print()
    print("Final team scores:")
    print(format_ranked_team_scores(team_scores))
    print()
    print(f"Winner: {winner.team_name} with {winner.score} points")


if __name__ == "__main__":
    main()
