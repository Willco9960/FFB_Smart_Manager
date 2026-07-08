from fantasy_engine.historical_loader import (
    DEFAULT_SEASON,
    get_player_stats_raw_path,
    get_player_stats_url,
    load_player_stats,
    read_player_stats,
)


def test_get_player_stats_url_uses_stats_player_release():
    url = get_player_stats_url(DEFAULT_SEASON)

    assert "stats_player" in url
    assert "2021" in url
    assert url.endswith(".csv")


def test_get_player_stats_raw_path_points_to_data_raw():
    path = get_player_stats_raw_path(DEFAULT_SEASON)

    assert str(path) == "data\\raw\\stats_player_week_2021.csv"


def test_read_player_stats_reads_local_csv(tmp_path):
    csv_path = tmp_path / "stats_player_week_2021.csv"
    csv_path.write_text(
        "player_id,player_name,season,week,recent_team,position,targets\n"
        "00-0000001,Test Player,2021,1,ATL,WR,8\n",
        encoding="utf-8",
    )

    rows = read_player_stats(csv_path)

    assert len(rows) == 1
    assert rows[0]["player_name"] == "Test Player"
    assert rows[0]["season"] == "2021"
    assert rows[0]["position"] == "WR"


def test_load_player_stats_reads_existing_local_file(tmp_path):
    csv_path = tmp_path / "stats_player_week_2021.csv"
    csv_path.write_text(
        "player_id,player_name,season,week,recent_team,position,targets\n"
        "00-0000002,Another Player,2021,1,DAL,RB,4\n",
        encoding="utf-8",
    )

    rows = load_player_stats(
        season=2021,
        raw_data_dir=tmp_path,
        force_download=False,
    )

    assert len(rows) == 1
    assert rows[0]["player_name"] == "Another Player"
