from dataclasses import dataclass

from agents.neural_lineup_agent import LineupAgent
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot
from fantasy_engine.season import (
    ESPN_TEN_TEAM_DEFAULT_RULES,
    ESPNLeagueRules,
    TeamStanding,
    rank_standings,
)
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from fantasy_engine.weekly_simulation import get_weekly_points_by_player, score_weekly_team


@dataclass(frozen=True)
class PlayoffGameResult:
    week: int
    first_seed: int
    second_seed: int
    first_team_score: float
    second_team_score: float
    winner_seed: int


@dataclass
class PlayoffSimulationResult:
    seeded_teams: list[Team]
    game_results: list[PlayoffGameResult]
    champion: Team


def get_playoff_teams(
    league: League,
    standings: dict[str, TeamStanding],
    rules: ESPNLeagueRules = ESPN_TEN_TEAM_DEFAULT_RULES,
) -> list[Team]:
    teams_by_name = {team.name: team for team in league.teams}
    ranked_standings = rank_standings(standings)

    return [
        teams_by_name[standing.team_name]
        for standing in ranked_standings[: rules.playoff_team_count]
    ]


def simulate_playoff_game(
    first_seed: int,
    second_seed: int,
    seeded_teams: list[Team],
    performances: list[WeeklyPlayerPerformance],
    week: int,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    lineup_agents: dict[str, LineupAgent] | None = None,
) -> PlayoffGameResult:
    weekly_points = get_weekly_points_by_player(performances, week)
    first_team = seeded_teams[first_seed - 1]
    second_team = seeded_teams[second_seed - 1]
    _, first_team_score = score_weekly_team(
        first_team,
        weekly_points,
        lineup_rules,
        None if lineup_agents is None else lineup_agents.get(first_team.name),
    )
    _, second_team_score = score_weekly_team(
        second_team,
        weekly_points,
        lineup_rules,
        None if lineup_agents is None else lineup_agents.get(second_team.name),
    )
    winner_seed = first_seed

    if second_team_score > first_team_score:
        winner_seed = second_seed

    return PlayoffGameResult(
        week=week,
        first_seed=first_seed,
        second_seed=second_seed,
        first_team_score=first_team_score,
        second_team_score=second_team_score,
        winner_seed=winner_seed,
    )


def simulate_espn_six_team_playoffs(
    league: League,
    standings: dict[str, TeamStanding],
    performances: list[WeeklyPlayerPerformance],
    rules: ESPNLeagueRules = ESPN_TEN_TEAM_DEFAULT_RULES,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    lineup_agents: dict[str, LineupAgent] | None = None,
) -> PlayoffSimulationResult:
    seeded_teams = get_playoff_teams(league, standings, rules)
    first_round_game_one = simulate_playoff_game(
        3, 6, seeded_teams, performances, 15, lineup_rules, lineup_agents
    )
    first_round_game_two = simulate_playoff_game(
        4, 5, seeded_teams, performances, 15, lineup_rules, lineup_agents
    )
    lowest_remaining_seed = max(first_round_game_one.winner_seed, first_round_game_two.winner_seed)
    other_remaining_seed = min(first_round_game_one.winner_seed, first_round_game_two.winner_seed)
    semifinal_game_one = simulate_playoff_game(
        1,
        lowest_remaining_seed,
        seeded_teams,
        performances,
        16,
        lineup_rules,
        lineup_agents,
    )
    semifinal_game_two = simulate_playoff_game(
        2,
        other_remaining_seed,
        seeded_teams,
        performances,
        16,
        lineup_rules,
        lineup_agents,
    )
    championship_game = simulate_playoff_game(
        semifinal_game_one.winner_seed,
        semifinal_game_two.winner_seed,
        seeded_teams,
        performances,
        17,
        lineup_rules,
        lineup_agents,
    )
    champion = seeded_teams[championship_game.winner_seed - 1]

    return PlayoffSimulationResult(
        seeded_teams=seeded_teams,
        game_results=[
            first_round_game_one,
            first_round_game_two,
            semifinal_game_one,
            semifinal_game_two,
            championship_game,
        ],
        champion=champion,
    )


def format_playoff_result(result: PlayoffSimulationResult) -> str:
    lines = ["Playoff results:"]

    for game_result in result.game_results:
        lines.append(
            f"Week {game_result.week}: Seed {game_result.first_seed} "
            f"({game_result.first_team_score:.2f}) vs Seed {game_result.second_seed} "
            f"({game_result.second_team_score:.2f}) -> Seed {game_result.winner_seed} advances"
        )

    lines.append(f"Champion: {result.champion.name}")

    return "\n".join(lines)
