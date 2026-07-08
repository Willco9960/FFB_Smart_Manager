from pathlib import Path

from fantasy_engine.processed_season import (
    build_processed_player_rows,
    get_processed_season_path,
    query_top_processed_players,
    rebuild_processed_season,
    write_processed_player_rows_to_parquet,
)


def test_get_processed_season_path_points_to_data_processed():
    path = get_processed_season_path(2021)

    assert str(path) == "data\\processed\\season_2021_player_scores.parquet"


def test_build_processed_player_rows_converts_raw_rows_to_clean_rows():
    raw_rows = [
        {
            "player_name": "Test QB",
            "position": "QB",
            "recent_team": "BUF",
            "passing_yards": "300",
            "passing_tds": "2",
            "interceptions": "1",
        }
    ]

    processed_rows = build_processed_player_rows(raw_rows)

    assert len(processed_rows) == 1
    assert processed_rows[0]["name"] == "Test QB"
    assert processed_rows[0]["position"] == "QB"
    assert processed_rows[0]["team"] == "BUF"
    assert processed_rows[0]["projected_score"] == 0.0
    assert processed_rows[0]["actual_score"] == 18.0


def test_write_processed_player_rows_to_parquet_and_query_with_duckdb(tmp_path):
    processed_rows = [
        {
            "name": "Lower Player",
            "position": "RB",
            "team": "ATL",
            "projected_score": 0.0,
            "actual_score": 10.0,
        },
        {
            "name": "Higher Player",
            "position": "WR",
            "team": "DAL",
            "projected_score": 0.0,
            "actual_score": 25.0,
        },
    ]
    parquet_path = tmp_path / "season_2021_player_scores.parquet"

    output_path = write_processed_player_rows_to_parquet(processed_rows, parquet_path)
    top_players = query_top_processed_players(output_path, limit=1)

    assert output_path.exists()
    assert top_players[0]["name"] == "Higher Player"
    assert top_players[0]["actual_score"] == 25.0


def test_rebuild_processed_season_from_existing_raw_file(tmp_path):
    raw_data_dir = tmp_path / "raw"
    processed_data_dir = tmp_path / "processed"
    raw_data_dir.mkdir()

    raw_csv_path = raw_data_dir / "stats_player_week_2021.csv"
    raw_csv_path.write_text(
        "player_name,position,recent_team,passing_yards,passing_tds,interceptions\n"
        "Historical QB,QB,KC,250,2,0\n",
        encoding="utf-8",
    )

    parquet_path = rebuild_processed_season(
        season=2021,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
    )
    top_players = query_top_processed_players(parquet_path, limit=1)

    assert parquet_path.exists()
    assert top_players[0]["name"] == "Historical QB"
    assert top_players[0]["actual_score"] == 18.0


def test_readme_documents_processed_rebuild_command():
    readme_text = Path("README.md").read_text(encoding="utf-8")

    assert "python -m scripts.rebuild_2021_processed_season" in readme_text
