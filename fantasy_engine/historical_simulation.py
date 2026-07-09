from dataclasses import dataclass
from pathlib import Path

import duckdb

from agents.random_agent import RandomDraftAgent
from fantasy_engine.draft import DraftPick, run_snake_draft
from fantasy_engine.historical_loader import DEFAULT_SEASON
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.processed_season import get_processed_season_path, rebuild_processed_season
from fantasy_engine.scoring import TeamScore, format_ranked_team_scores, get_winner, score_teams
from fantasy_engine.team import Team

DEFAULT_TEAM_COUNT = 10
DEFAULT_ROSTER_SIZE = 16


@dataclass
class HistoricalSimulationResult:
    league: League
    draft_results: list[DraftPick]
    team_scores: list[TeamScore]
    winner: TeamScore


def load_processed_player_pool(parquet_path: Path) -> list[Player]:
    connection = duckdb.connect()
    rows = connection.execute(
        """
        SELECT name, position, team, projected_score, actual_score
        FROM read_parquet(?)
        WHERE position IN ('QB', 'RB', 'WR', 'TE')
        ORDER BY actual_score DESC
        """,
        [str(parquet_path)],
    ).fetchall()
    connection.close()

    players = []

    for row in rows:
        team = ""

        if row[2] is not None:
            team = str(row[2])

        player = Player(
            name=str(row[0]),
            position=str(row[1]),
            team=team,
            projected_score=float(row[3]),
            actual_score=float(row[4]),
        )
        players.append(player)

    return players


def create_historical_league(
    players: list[Player],
    team_count: int = DEFAULT_TEAM_COUNT,
) -> League:
    teams = []

    for number in range(1, team_count + 1):
        teams.append(Team(name=f"Historical Team {number}"))

    return League(
        name="2021 Historical Simulation League",
        teams=teams,
        available_players=players,
    )


def run_historical_draft_simulation(
    season: int = DEFAULT_SEASON,
    parquet_path: Path | None = None,
    team_count: int = DEFAULT_TEAM_COUNT,
    roster_size: int = DEFAULT_ROSTER_SIZE,
    seed: int = 2021,
    rebuild_if_missing: bool = True,
) -> HistoricalSimulationResult:
    if parquet_path is None:
        parquet_path = get_processed_season_path(season)

    if rebuild_if_missing and not parquet_path.exists():
        parquet_path = rebuild_processed_season(season)

    players = load_processed_player_pool(parquet_path)
    required_player_count = team_count * roster_size

    if len(players) < required_player_count:
        raise ValueError(
            f"Historical player pool has {len(players)} players, "
            f"but the draft needs {required_player_count} players."
        )

    league = create_historical_league(players, team_count)
    draft_agent = RandomDraftAgent(seed=seed)
    draft_results = run_snake_draft(
        league=league,
        rounds=roster_size,
        draft_agent=draft_agent,
    )
    team_scores = score_teams(league.teams)
    winner = get_winner(team_scores)

    return HistoricalSimulationResult(
        league=league,
        draft_results=draft_results,
        team_scores=team_scores,
        winner=winner,
    )


def format_historical_simulation_summary(result: HistoricalSimulationResult) -> str:
    return (
        "Historical draft simulation complete\n"
        f"Teams: {result.league.team_count()}\n"
        f"Draft picks: {len(result.draft_results)}\n"
        "\n"
        "Final historical team scores:\n"
        f"{format_ranked_team_scores(result.team_scores)}\n"
        "\n"
        f"Historical season winner: "
        f"{result.winner.team_name} with {result.winner.score} points"
    )
