from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES
from fantasy_engine.player import Player
from fantasy_engine.season import ScheduledMatchup, initialize_standings
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from fantasy_engine.weekly_simulation import (
    create_weekly_scored_roster,
    score_weekly_team,
    simulate_historical_week,
)


def create_player(name: str, position: str, projection: float) -> Player:
    return Player(name=name, position=position, team="ATL", projected_score=projection)


def create_complete_offensive_roster(team_name: str) -> Team:
    return Team(
        name=team_name,
        roster=[
            create_player(f"{team_name} QB", "QB", 20.0),
            create_player(f"{team_name} RB 1", "RB", 18.0),
            create_player(f"{team_name} RB 2", "RB", 17.0),
            create_player(f"{team_name} WR 1", "WR", 16.0),
            create_player(f"{team_name} WR 2", "WR", 15.0),
            create_player(f"{team_name} TE", "TE", 14.0),
            create_player(f"{team_name} FLEX", "WR", 13.0),
            create_player(f"{team_name} Bench QB", "QB", 1.0),
        ],
    )


def create_weekly_performances(team: Team, points: float) -> list[WeeklyPlayerPerformance]:
    return [
        WeeklyPlayerPerformance(
            player_id=player.name,
            player_name=player.name,
            position=player.position,
            team=player.team,
            week=1,
            fantasy_points=points,
        )
        for player in team.roster
    ]


def test_create_weekly_scored_roster_applies_only_current_week_points():
    roster = [create_player("Test RB", "RB", 15.0)]

    weekly_roster = create_weekly_scored_roster(roster, {("Test RB", "RB"): 22.0})

    assert weekly_roster[0].projected_score == 15.0
    assert weekly_roster[0].actual_score == 22.0


def test_score_weekly_team_uses_projection_for_lineup_selection():
    team = create_complete_offensive_roster("Team 1")
    weekly_points = {(player.name, player.position): 10.0 for player in team.roster}
    weekly_points[("Team 1 Bench QB", "QB")] = 30.0

    starting_lineup, team_score = score_weekly_team(team, weekly_points)

    assert "Team 1 QB" in [player.name for player in starting_lineup.players]
    assert "Team 1 Bench QB" not in [player.name for player in starting_lineup.players]
    assert team_score == 70.0


def test_simulate_historical_week_updates_head_to_head_standings():
    first_team = create_complete_offensive_roster("Team 1")
    second_team = create_complete_offensive_roster("Team 2")
    standings = initialize_standings([first_team.name, second_team.name])
    schedule = [ScheduledMatchup(week=1, first_team_name="Team 1", second_team_name="Team 2")]
    performances = [
        *create_weekly_performances(first_team, 20.0),
        *create_weekly_performances(second_team, 10.0),
    ]

    weekly_scores = simulate_historical_week(
        teams=[first_team, second_team],
        standings=standings,
        schedule=schedule,
        performances=performances,
        week=1,
        lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
    )

    assert weekly_scores["Team 1"] == 140.0
    assert weekly_scores["Team 2"] == 70.0
    assert standings["Team 1"].wins == 1
    assert standings["Team 2"].losses == 1
