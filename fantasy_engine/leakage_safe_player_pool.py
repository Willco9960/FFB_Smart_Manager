from fantasy_engine.fantasy_points import (
    STANDARD_SCORING,
    FantasyScoringSettings,
    calculate_fantasy_points,
)
from fantasy_engine.historical_loader import RAW_DATA_DIR, load_player_stats
from fantasy_engine.historical_player_pool import (
    FANTASY_RELEVANT_POSITIONS,
    SPECIAL_TEAMS_POSITIONS,
    create_team_defense_rows,
    get_player_id,
    get_player_name,
    get_player_position,
    get_player_team,
)
from fantasy_engine.player import Player


def create_player_key(row: dict[str, str]) -> tuple[str, str]:
    return (
        get_player_id(row),
        get_player_position(row),
    )


def is_valid_fantasy_row(row: dict[str, str]) -> bool:
    player_name = get_player_name(row)
    position = get_player_position(row)

    return player_name != "" and position in FANTASY_RELEVANT_POSITIONS


def is_valid_extended_fantasy_row(row: dict[str, str]) -> bool:
    player_name = get_player_name(row)
    position = get_player_position(row)

    return (
        player_name != ""
        and position in FANTASY_RELEVANT_POSITIONS | SPECIAL_TEAMS_POSITIONS | {"DST"}
    )


def build_season_totals(
    rows: list[dict[str, str]],
    scoring_settings: FantasyScoringSettings = STANDARD_SCORING,
    include_special_teams: bool = False,
) -> dict[tuple[str, str], dict[str, str | float]]:
    season_totals: dict[tuple[str, str], dict[str, str | float | bool]] = {}

    source_rows = list(rows)

    if include_special_teams:
        source_rows.extend(create_team_defense_rows(rows))

    for row in source_rows:
        valid_row = (
            is_valid_extended_fantasy_row(row)
            if include_special_teams
            else is_valid_fantasy_row(row)
        )

        if not valid_row:
            continue

        player_key = create_player_key(row)
        fantasy_points = calculate_fantasy_points(row, scoring_settings)

        if player_key not in season_totals:
            season_totals[player_key] = {
                "player_id": get_player_id(row),
                "name": get_player_name(row),
                "position": get_player_position(row),
                "team": get_player_team(row),
                "fantasy_points": 0.0,
            }

        season_totals[player_key]["fantasy_points"] = round(
            float(season_totals[player_key]["fantasy_points"]) + fantasy_points,
            2,
        )

        team = get_player_team(row)

        if team != "":
            season_totals[player_key]["team"] = team

    return season_totals


def create_leakage_safe_player_pool(
    projection_rows: list[dict[str, str]],
    actual_rows: list[dict[str, str]],
    scoring_settings: FantasyScoringSettings = STANDARD_SCORING,
    include_special_teams: bool = False,
    include_actual_only: bool = False,
) -> list[Player]:
    projection_totals = build_season_totals(
        rows=projection_rows,
        scoring_settings=scoring_settings,
        include_special_teams=include_special_teams,
    )
    actual_totals = build_season_totals(
        rows=actual_rows,
        scoring_settings=scoring_settings,
        include_special_teams=include_special_teams,
    )
    players = []

    player_keys = (
        projection_totals.keys() | actual_totals.keys()
        if include_actual_only
        else projection_totals.keys()
    )
    for player_key in player_keys:
        projection_player = projection_totals.get(player_key)
        actual_player = actual_totals.get(player_key)
        # Projection-season metadata is the pre-season identity/team view.
        # Target-season rows supply outcomes only; using their metadata would
        # leak post-season roster changes into the draft universe.
        source_player = projection_player or actual_player
        if source_player is None:
            continue

        player = Player(
            name=str(source_player["name"]),
            position=str(source_player["position"]),
            team=str(source_player["team"]),
            projected_score=(
                float(projection_player["fantasy_points"])
                if projection_player is not None
                else 0.0
            ),
            actual_score=(
                float(actual_player["fantasy_points"])
                if actual_player is not None
                else 0.0
            ),
            player_id=str(source_player["player_id"]),
            history_missing=projection_player is None,
        )
        players.append(player)

    return sorted(
        players,
        key=lambda player: player.projected_score,
        reverse=True,
    )


def load_leakage_safe_player_pool(
    projection_season: int = 2020,
    actual_season: int = 2021,
    raw_data_dir=RAW_DATA_DIR,
    scoring_settings: FantasyScoringSettings = STANDARD_SCORING,
    include_special_teams: bool = False,
    include_actual_only: bool = False,
) -> list[Player]:
    projection_rows = load_player_stats(
        season=projection_season,
        raw_data_dir=raw_data_dir,
    )
    actual_rows = load_player_stats(
        season=actual_season,
        raw_data_dir=raw_data_dir,
    )

    return create_leakage_safe_player_pool(
        projection_rows=projection_rows,
        actual_rows=actual_rows,
        scoring_settings=scoring_settings,
        include_special_teams=include_special_teams,
        include_actual_only=include_actual_only,
    )
