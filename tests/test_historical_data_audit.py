from fantasy_engine.historical_loader import validate_player_stats_schema


def test_validate_player_stats_schema_accepts_required_columns(tmp_path):
    path = tmp_path / "stats.csv"
    path.write_text(
        "player_id,player_name,position,season,week,recent_team,opponent_team,"
        "targets,carries,attempts,passing_yards,rushing_yards,receiving_yards,"
        "def_sacks,def_interceptions,def_tds,fg_made_0_19,pat_made\n",
        encoding="utf-8",
    )

    header = validate_player_stats_schema(path)

    assert "player_id" in header


def test_validate_player_stats_schema_rejects_missing_columns(tmp_path):
    path = tmp_path / "stats.csv"
    path.write_text("player_id,player_name,position\n", encoding="utf-8")

    try:
        validate_player_stats_schema(path)
    except ValueError as error:
        assert "missing columns" in str(error)
    else:
        raise AssertionError("Expected missing-column validation to fail")
