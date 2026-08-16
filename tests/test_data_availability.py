import pytest

from fantasy_engine.data_availability import inspect_rows, validate_training_seasons


def test_data_manifest_records_optional_context_missingness():
    rows = [
        {
            "player_id": "p1",
            "player_name": "Player",
            "position": "WR",
            "season": "2024",
            "week": "1",
            "recent_team": "ATL",
        }
    ]
    manifest = inspect_rows(2024, rows, "fixture.csv")

    manifest.require()
    assert "game_total" in manifest.missing_optional_columns
    assert manifest.coverage_for("player_id") == 1.0


def test_data_manifest_rejects_missing_core_columns():
    manifest = inspect_rows(
        2024,
        [{"player_name": "Player", "position": "WR"}],
        "fixture.csv",
    )

    with pytest.raises(ValueError, match="not training-ready"):
        manifest.require()


def test_training_season_validation_requires_real_schema(tmp_path):
    path = tmp_path / "stats_player_week_2024.csv"
    path.write_text(
        "player_id,player_name,position,season,week,recent_team,opponent_team,targets,carries,attempts,"
        "passing_yards,rushing_yards,receiving_yards,def_sacks,def_interceptions,def_tds,"
        "fg_made_0_19,pat_made\n"
        "p1,Player,WR,2024,1,ATL,TB,4,0,0,0,0,40,0,0,0,0,0\n",
        encoding="utf-8",
    )

    manifests = validate_training_seasons((2024,), raw_data_dir=tmp_path)

    assert manifests[0].season == 2024
