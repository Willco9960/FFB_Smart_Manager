from fantasy_engine.fantasy_points import (
    STANDARD_SCORING,
    FantasyScoringSettings,
    calculate_fantasy_points,
)
from fantasy_engine.player import Player

FANTASY_RELEVANT_POSITIONS = {"QB", "RB", "WR", "TE"}
SPECIAL_TEAMS_POSITIONS = {"K"}
DEFENSIVE_POSITIONS = {
    "CB",
    "DB",
    "DE",
    "DL",
    "DT",
    "FS",
    "ILB",
    "LB",
    "MLB",
    "NT",
    "OLB",
    "S",
    "SAF",
}


def is_fantasy_relevant_row(row: dict[str, str]) -> bool:
    position = row.get("position", "")

    return position in FANTASY_RELEVANT_POSITIONS


def get_player_name(row: dict[str, str]) -> str:
    if row.get("position", "") == "DST":
        return f"{get_player_team(row)} D/ST"

    return row.get("player_name", "")


def get_player_position(row: dict[str, str]) -> str:
    if row.get("position", "") in DEFENSIVE_POSITIONS:
        return "DST"

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


def create_team_defense_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    defensive_totals: dict[tuple[str, str], dict[str, str]] = {}
    defensive_stat_names = (
        "def_sacks",
        "def_interceptions",
        "fumble_recovery_opp",
        "def_tds",
        "def_safeties",
    )

    for row in rows:
        if row.get("position", "") not in DEFENSIVE_POSITIONS:
            continue

        team = get_player_team(row)
        week = row.get("week", "")
        key = (team, week)

        if key not in defensive_totals:
            defensive_totals[key] = {
                "player_name": f"{team} D/ST",
                "position": "DST",
                "recent_team": team,
                "week": week,
            }

            for stat_name in defensive_stat_names:
                defensive_totals[key][stat_name] = "0"

        for stat_name in defensive_stat_names:
            current_value = float(defensive_totals[key][stat_name])
            row_value = float(row.get(stat_name, "0") or "0")
            defensive_totals[key][stat_name] = str(current_value + row_value)

    return list(defensive_totals.values())
