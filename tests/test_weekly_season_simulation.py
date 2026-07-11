from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from fantasy_engine.weekly_season_simulation import (
    format_final_standings,
    run_historical_regular_season,
)


def create_complete_team(team_number: int) -> Team:
    team_name = f"Team {team_number}"
    positions = ["QB", "RB", "RB", "WR", "WR", "TE", "WR"]
    roster = []

    for index, position in enumerate(positions, start=1):
        roster.append(
            Player(
                name=f"{team_name} {position} {index}",
                position=position,
                team="ATL",
                projected_score=20.0,
            )
        )

    return Team(name=team_name, roster=roster)


def create_performances(teams: list[Team]) -> list[WeeklyPlayerPerformance]:
    performances = []

    for week in range(1, 15):
        for team_number, team in enumerate(teams, start=1):
            for player in team.roster:
                performances.append(
                    WeeklyPlayerPerformance(
                        player_id=player.name,
                        player_name=player.name,
                        position=player.position,
                        team=player.team,
                        week=week,
                        fantasy_points=float(team_number),
                    )
                )

    return performances


def test_run_historical_regular_season_simulates_fourteen_weeks():
    teams = [create_complete_team(number) for number in range(1, 11)]
    league = League(name="Test League", teams=teams)
    performances = create_performances(teams)

    result = run_historical_regular_season(league, performances)

    assert len(result.weekly_scores) == 14
    assert all(len(weekly_scores) == 10 for weekly_scores in result.weekly_scores.values())
    assert all(
        standing.wins + standing.losses + standing.ties == 14
        for standing in result.standings.values()
    )


def test_final_standings_sort_highest_scoring_winning_team_first():
    teams = [create_complete_team(number) for number in range(1, 11)]
    league = League(name="Test League", teams=teams)
    performances = create_performances(teams)

    result = run_historical_regular_season(league, performances)
    standings_text = format_final_standings(result)

    assert result.ranked_standings()[0].team_name == "Team 10"
    assert "Final regular-season standings:" in standings_text
