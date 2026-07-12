from dataclasses import replace

from agents.neural_lineup_agent import LineupAgent
from fantasy_engine.lineup import (
    ESPN_OFFENSIVE_LINEUP_RULES,
    LineupSlot,
    StartingLineup,
    build_best_starting_lineup,
)
from fantasy_engine.player import Player
from fantasy_engine.season import ScheduledMatchup, TeamStanding, record_matchup_result
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from fantasy_engine.weekly_projection import create_weekly_projected_roster


def get_weekly_points_by_player(
    performances: list[WeeklyPlayerPerformance],
    week: int,
) -> dict[tuple[str, str], float]:
    return {
        (performance.player_name, performance.position): performance.fantasy_points
        for performance in performances
        if performance.week == week
    }


def create_weekly_scored_roster(
    roster: list[Player],
    weekly_points_by_player: dict[tuple[str, str], float],
) -> list[Player]:
    return [
        replace(
            player,
            actual_score=weekly_points_by_player.get((player.name, player.position), 0.0),
        )
        for player in roster
    ]


def select_projected_weekly_lineup(
    roster: list[Player],
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
) -> StartingLineup:
    return build_best_starting_lineup(
        roster=roster,
        lineup_rules=lineup_rules,
        selection_score_attribute="projected_score",
    )


def score_weekly_team(
    team: Team,
    weekly_points_by_player: dict[tuple[str, str], float],
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    lineup_agent: LineupAgent | None = None,
) -> tuple[StartingLineup, float]:
    weekly_scored_roster = create_weekly_scored_roster(team.roster, weekly_points_by_player)
    if lineup_agent is None:
        starting_lineup = select_projected_weekly_lineup(weekly_scored_roster, lineup_rules)
    else:
        starting_lineup = lineup_agent.choose_lineup(weekly_scored_roster)

    return starting_lineup, starting_lineup.score()


def score_adaptive_weekly_team(
    team: Team,
    performances: list[WeeklyPlayerPerformance],
    week: int,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    lineup_agent: LineupAgent | None = None,
) -> tuple[StartingLineup, float]:
    weekly_points_by_player = get_weekly_points_by_player(performances, week)
    projected_roster = create_weekly_projected_roster(team.roster, performances, week)
    projected_team = Team(name=team.name, roster=projected_roster)

    return score_weekly_team(
        projected_team,
        weekly_points_by_player,
        lineup_rules,
        lineup_agent,
    )


def simulate_historical_week(
    teams: list[Team],
    standings: dict[str, TeamStanding],
    schedule: list[ScheduledMatchup],
    performances: list[WeeklyPlayerPerformance],
    week: int,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    lineup_agents: dict[str, LineupAgent] | None = None,
) -> dict[str, float]:
    teams_by_name = {team.name: team for team in teams}
    weekly_scores = {}

    for matchup in schedule:
        if matchup.week != week:
            continue

        first_team = teams_by_name[matchup.first_team_name]
        second_team = teams_by_name[matchup.second_team_name]
        _, first_team_score = score_adaptive_weekly_team(
            first_team,
            performances,
            week,
            lineup_rules,
            None if lineup_agents is None else lineup_agents.get(first_team.name),
        )
        _, second_team_score = score_adaptive_weekly_team(
            second_team,
            performances,
            week,
            lineup_rules,
            None if lineup_agents is None else lineup_agents.get(second_team.name),
        )
        weekly_scores[first_team.name] = first_team_score
        weekly_scores[second_team.name] = second_team_score
        record_matchup_result(standings, matchup, first_team_score, second_team_score)

    return weekly_scores
