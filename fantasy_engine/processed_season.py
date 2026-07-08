import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from fantasy_engine.fantasy_points import STANDARD_SCORING, FantasyScoringSettings
from fantasy_engine.historical_loader import DEFAULT_SEASON, RAW_DATA_DIR, load_player_stats
from fantasy_engine.historical_player_pool import create_historical_player_pool

PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_PLAYER_FIELDS = [
    "name",
    "position",
    "team",
    "projected_score",
    "actual_score",
]


def get_processed_season_path(
    season: int = DEFAULT_SEASON,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> Path:
    return processed_data_dir / f"season_{season}_player_scores.parquet"


def build_processed_player_rows(
    raw_rows: list[dict[str, str]],
    scoring_settings: FantasyScoringSettings = STANDARD_SCORING,
) -> list[dict[str, str | float]]:
    players = create_historical_player_pool(raw_rows, scoring_settings)
    processed_rows = []

    for player in players:
        processed_rows.append(
            {
                "name": player.name,
                "position": player.position,
                "team": player.team,
                "projected_score": player.projected_score,
                "actual_score": player.actual_score,
            }
        )

    return processed_rows


def write_processed_player_rows_to_parquet(
    processed_rows: list[dict[str, str | float]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        temp_csv_path = Path(temp_dir) / "processed_players.csv"

        with temp_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=PROCESSED_PLAYER_FIELDS)
            writer.writeheader()
            writer.writerows(processed_rows)

        connection = duckdb.connect()
        connection.execute(
            "CREATE TABLE processed_players AS SELECT * FROM read_csv_auto(?, header=true)",
            [str(temp_csv_path)],
        )
        connection.execute(
            "COPY processed_players TO ? (FORMAT parquet)",
            [str(output_path)],
        )
        connection.close()

    return output_path


def query_top_processed_players(
    parquet_path: Path,
    limit: int = 10,
) -> list[dict[str, object]]:
    connection = duckdb.connect()
    rows = connection.execute(
        """
        SELECT name, position, team, actual_score
        FROM read_parquet(?)
        ORDER BY actual_score DESC
        LIMIT ?
        """,
        [str(parquet_path), limit],
    ).fetchall()
    connection.close()

    results = []

    for row in rows:
        results.append(
            {
                "name": row[0],
                "position": row[1],
                "team": row[2],
                "actual_score": row[3],
            }
        )

    return results


def rebuild_processed_season(
    season: int = DEFAULT_SEASON,
    raw_data_dir: Path = RAW_DATA_DIR,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    scoring_settings: FantasyScoringSettings = STANDARD_SCORING,
) -> Path:
    raw_rows = load_player_stats(season=season, raw_data_dir=raw_data_dir)
    processed_rows = build_processed_player_rows(raw_rows, scoring_settings)
    output_path = get_processed_season_path(season, processed_data_dir)

    return write_processed_player_rows_to_parquet(processed_rows, output_path)
