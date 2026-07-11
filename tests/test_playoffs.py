from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.playoffs import (
    format_playoff_result,
    get_playoff_teams,
    simulate_espn_six_team_playoffs,
)
from fantasy_engine.season import initialize_standings
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import WeeklyPlayerPerformance


def create_complete_team(team_number: int) -> Team:
    team_name = f"Team {team_number}"
    positions = ["QB", "RB", "RB", "WR", "WR", "TE", "WR"]
    roster = [
        Player(
            name=f"{team_name} {position} {index}",
            position=position,
            team="ATL",
            projected_score=20.0,
        )
        for index, position in enumerate(positions, start=1)
    ]

    return Team(name=team_name, roster=roster)


def create_playoff_performances(teams: list[Team]) -> list[WeeklyPlayerPerformance]:
    performances = []

    for week in (15, 16, 17):
        for team_number, team in enumerate(teams, start=1):
            for player in team.roster:
                performances.append(
                    WeeklyPlayerPerformance(
                        player_id=player.name,
                        player_name=player.name,
                        position=player.position,
                        team=player.team,
                        week=week,
                        fantasy_points=float(11 - team_number),
                    )
                )

    return performances


def create_ranked_standings(team_names: list[str]):
    standings = initialize_standings(team_names)

    for index, team_name in enumerate(team_names, start=1):
        standings[team_name].wins = 11 - index
        standings[team_name].points_for = float(1100 - (index * 10))

    return standings


def test_get_playoff_teams_uses_top_six_seeds():
    teams = [create_complete_team(number) for number in range(1, 11)]
    league = League(name="Test League", teams=teams)
    standings = create_ranked_standings([team.name for team in teams])

    playoff_teams = get_playoff_teams(league, standings)

    assert [team.name for team in playoff_teams] == [f"Team {number}" for number in range(1, 7)]


def test_simulate_espn_six_team_playoffs_advances_highest_scoring_seed():
    teams = [create_complete_team(number) for number in range(1, 11)]
    league = League(name="Test League", teams=teams)
    standings = create_ranked_standings([team.name for team in teams])
    performances = create_playoff_performances(teams)

    result = simulate_espn_six_team_playoffs(league, standings, performances)

    assert len(result.game_results) == 5
    assert result.champion.name == "Team 1"
    assert result.game_results[0].first_seed == 3
    assert result.game_results[0].second_seed == 6


def test_format_playoff_result_includes_champion():
    teams = [create_complete_team(number) for number in range(1, 11)]
    league = League(name="Test League", teams=teams)
    standings = create_ranked_standings([team.name for team in teams])
    performances = create_playoff_performances(teams)

    result = simulate_espn_six_team_playoffs(league, standings, performances)

    assert "Champion: Team 1" in format_playoff_result(result)
