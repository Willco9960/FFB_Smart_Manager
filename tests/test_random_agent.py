from agents.random_agent import RandomDraftAgent
from fantasy_engine.draft import run_snake_draft
from fantasy_engine.fake_data import create_fake_player_pool
from fantasy_engine.league import League
from fantasy_engine.team import Team


def create_test_league(team_count: int = 10) -> League:
    teams = []

    for number in range(1, team_count + 1):
        teams.append(Team(name=f"Team {number}"))

    return League(
        name="Random Agent Test League",
        teams=teams,
        available_players=create_fake_player_pool(),
    )


def test_random_agent_chooses_from_available_players():
    league = create_test_league()
    agent = RandomDraftAgent(seed=42)

    selected_player = agent.choose_player(
        available_players=league.available_players,
        team=league.teams[0],
        league=league,
    )

    assert selected_player in league.available_players


def test_random_agent_draft_completes_without_errors():
    league = create_test_league()
    agent = RandomDraftAgent(seed=42)

    draft_results = run_snake_draft(
        league=league,
        rounds=16,
        draft_agent=agent,
    )

    assert len(draft_results) == 160

    for team in league.teams:
        assert team.roster_size() == 16


def test_random_agent_does_not_draft_same_player_twice():
    league = create_test_league()
    agent = RandomDraftAgent(seed=42)

    draft_results = run_snake_draft(
        league=league,
        rounds=16,
        draft_agent=agent,
    )

    drafted_player_names = [pick.player.name for pick in draft_results]

    assert len(drafted_player_names) == len(set(drafted_player_names))
