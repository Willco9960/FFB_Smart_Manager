from fantasy_engine.fantasy_points import (
    STANDARD_SCORING,
    FantasyScoringSettings,
    calculate_fantasy_points,
)
from fantasy_engine.player import Player

FANTASY_RELEVANT_POSITIONS = {"QB", "RB", "WR", "TE"}


def is_fantasy_relevant_row(row: dict[str, str]) -> bool:
    position = row.get("position", "")

    return position in FANTASY_RELEVANT_POSITIONS


def get_player_name(row: dict[str, str]) -> str:
    return row.get("player_name", "")


def get_player_position(row: dict[str, str]) -> str:
    return row.get("position", "")


def get_player_team(row: dict[str, str]) -> str:
    team = row.get("recent_team", "")

    if team in ("", "None"):
        team = row.get("team", "")

    if team in ("", "None"):
        team = row.get("posteam", "")

    return team


def create_player_from_historical_row(
    row: dict[str, str],
    scoring_settings: FantasyScoringSettings = STANDARD_SCORING,
) -> Player:
    fantasy_points = calculate_fantasy_points(row, scoring_settings)

    return Player(
        name=get_player_name(row),
        position=get_player_position(row),
        team=get_player_team(row),
        projected_score=0.0,
        actual_score=fantasy_points,
    )


def create_historical_player_pool(
    rows: list[dict[str, str]],
    scoring_settings: FantasyScoringSettings = STANDARD_SCORING,
) -> list[Player]:
    players = []

    for row in rows:
        if is_fantasy_relevant_row(row):
            player = create_player_from_historical_row(row, scoring_settings)
            players.append(player)

    return players
