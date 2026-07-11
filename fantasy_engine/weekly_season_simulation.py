from dataclasses import dataclass, replace

from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot
from fantasy_engine.season import (
    ESPN_TEN_TEAM_DEFAULT_RULES,
    ESPNLeagueRules,
    ScheduledMatchup,
    TeamStanding,
    create_regular_season_schedule,
    initialize_standings,
    rank_standings,
)
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from fantasy_engine.weekly_simulation import simulate_historical_week


@dataclass
class RegularSeasonSimulationResult:
    league: League
    schedule: list[ScheduledMatchup]
    standings: dict[str, TeamStanding]
    weekly_scores: dict[int, dict[str, float]]
    weekly_standings: dict[int, list[TeamStanding]]

    def ranked_standings(self) -> list[TeamStanding]:
        return rank_standings(self.standings)


def run_historical_regular_season(
    league: League,
    performances: list[WeeklyPlayerPerformance],
    rules: ESPNLeagueRules = ESPN_TEN_TEAM_DEFAULT_RULES,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
) -> RegularSeasonSimulationResult:
    team_names = [team.name for team in league.teams]
    schedule = create_regular_season_schedule(team_names, rules)
    standings = initialize_standings(team_names)
    weekly_scores = {}
    weekly_standings = {}

    for week in range(1, rules.regular_season_weeks + 1):
        weekly_scores[week] = simulate_historical_week(
            teams=league.teams,
            standings=standings,
            schedule=schedule,
            performances=performances,
            week=week,
            lineup_rules=lineup_rules,
        )
        weekly_standings[week] = [replace(standing) for standing in rank_standings(standings)]

    return RegularSeasonSimulationResult(
        league=league,
        schedule=schedule,
        standings=standings,
        weekly_scores=weekly_scores,
        weekly_standings=weekly_standings,
    )


def format_final_standings(result: RegularSeasonSimulationResult) -> str:
    lines = ["Final regular-season standings:"]

    for rank, standing in enumerate(result.ranked_standings(), start=1):
        lines.append(
            f"{rank}. {standing.team_name}: "
            f"{standing.wins}-{standing.losses}-{standing.ties}, "
            f"PF {standing.points_for:.2f}"
        )

    return "\n".join(lines)


def format_week_by_week_report(result: RegularSeasonSimulationResult) -> str:
    lines = []

    for week, weekly_scores in result.weekly_scores.items():
        lines.append(f"Week {week} results:")

        for matchup in result.schedule:
            if matchup.week != week:
                continue

            lines.append(
                f"{matchup.first_team_name} {weekly_scores[matchup.first_team_name]:.2f} "
                f"vs {matchup.second_team_name} {weekly_scores[matchup.second_team_name]:.2f}"
            )

        lines.append(f"Standings after Week {week}:")

        for rank, standing in enumerate(result.weekly_standings[week], start=1):
            lines.append(
                f"{rank}. {standing.team_name}: "
                f"{standing.wins}-{standing.losses}-{standing.ties}, "
                f"PF {standing.points_for:.2f}"
            )

        lines.append("")

    return "\n".join(lines)
