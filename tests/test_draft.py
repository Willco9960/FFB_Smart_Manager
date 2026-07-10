from fantasy_engine.draft import (
    format_draft_results,
    get_snake_draft_order,
    run_snake_draft,
)
from fantasy_engine.fake_data import create_fake_player_pool
from fantasy_engine.league import League
from fantasy_engine.team import Team


class FirstAvailableAgent:
    def choose_player(self, available_players, team, league):
        return available_players[0]


class LastAvailableAgent:
    def choose_player(self, available_players, team, league):
        return available_players[-1]


def create_test_league(team_count: int = 10) -> League:
    teams = []

    for number in range(1, team_count + 1):
        teams.append(Team(name=f"Team {number}"))

    return League(
        name="Test League",
        teams=teams,
        available_players=create_fake_player_pool(),
    )


def test_snake_draft_order_alternates_each_round():
    teams = [Team(name=f"Team {number}") for number in range(1, 4)]

    round_one_order = get_snake_draft_order(teams, 1)
    round_two_order = get_snake_draft_order(teams, 2)
    round_three_order = get_snake_draft_order(teams, 3)

    assert [team.name for team in round_one_order] == ["Team 1", "Team 2", "Team 3"]
    assert [team.name for team in round_two_order] == ["Team 3", "Team 2", "Team 1"]
    assert [team.name for team in round_three_order] == ["Team 1", "Team 2", "Team 3"]


def test_snake_draft_gives_each_team_expected_roster_size():
    league = create_test_league()

    run_snake_draft(league, rounds=16)

    for team in league.teams:
        assert team.roster_size() == 16


def test_snake_draft_does_not_draft_same_player_twice():
    league = create_test_league()

    draft_results = run_snake_draft(league, rounds=16)
    drafted_player_names = [pick.player.name for pick in draft_results]

    assert len(drafted_player_names) == len(set(drafted_player_names))


def test_snake_draft_creates_expected_number_of_picks():
    league = create_test_league()

    draft_results = run_snake_draft(league, rounds=16)

    assert len(draft_results) == 160


def test_draft_results_can_be_formatted():
    league = create_test_league()

    draft_results = run_snake_draft(league, rounds=1)
    formatted_results = format_draft_results(draft_results)

    assert "Pick 1:" in formatted_results
    assert "Round 1" in formatted_results
    assert "selected" in formatted_results


def test_snake_draft_uses_the_agent_assigned_to_each_team():
    league = create_test_league(team_count=2)
    team_agents = {
        "Team 1": FirstAvailableAgent(),
        "Team 2": LastAvailableAgent(),
    }

    draft_results = run_snake_draft(
        league=league,
        rounds=1,
        team_agents=team_agents,
    )

    assert draft_results[0].player.name == "QB Player 1"
    assert draft_results[1].player.name == "DST Player 16"
