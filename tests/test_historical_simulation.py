from fantasy_engine.historical_simulation import (
    create_historical_league,
    format_historical_simulation_summary,
    format_winning_team_draft_picks,
    load_processed_player_pool,
    run_historical_draft_simulation,
)
from fantasy_engine.player import Player
from fantasy_engine.processed_season import write_processed_player_rows_to_parquet


def test_load_processed_player_pool_reads_parquet_players(tmp_path):
    processed_rows = [
        {
            "name": "Historical RB",
            "position": "RB",
            "team": "IND",
            "projected_score": 0.0,
            "actual_score": 20.5,
        }
    ]
    parquet_path = tmp_path / "season_2021_player_scores.parquet"
    write_processed_player_rows_to_parquet(processed_rows, parquet_path)

    players = load_processed_player_pool(parquet_path)

    assert len(players) == 1
    assert players[0].name == "Historical RB"
    assert players[0].position == "RB"
    assert players[0].team == "IND"
    assert players[0].actual_score == 20.5


def test_create_historical_league_uses_requested_team_count():
    players = [
        Player(
            name="Historical Player",
            position="WR",
            team="CIN",
            actual_score=10.0,
        )
    ]

    league = create_historical_league(players, team_count=4)

    assert league.name == "2021 Historical Simulation League"
    assert league.team_count() == 4
    assert league.available_player_count() == 1


def test_run_historical_draft_simulation_scores_and_declares_winner(tmp_path):
    processed_rows = []

    for number in range(1, 21):
        processed_rows.append(
            {
                "name": f"Historical Player {number}",
                "position": "RB",
                "team": "ATL",
                "projected_score": 0.0,
                "actual_score": float(number),
            }
        )

    parquet_path = tmp_path / "season_2021_player_scores.parquet"
    write_processed_player_rows_to_parquet(processed_rows, parquet_path)

    result = run_historical_draft_simulation(
        parquet_path=parquet_path,
        team_count=4,
        roster_size=2,
        seed=2021,
        rebuild_if_missing=False,
    )

    assert len(result.draft_results) == 8
    assert len(result.team_scores) == 4
    assert result.winner.score > 0.0
    assert result.winner.team_name.startswith("Historical Team")


def test_format_historical_simulation_summary_includes_winner(tmp_path):
    processed_rows = []

    for number in range(1, 21):
        processed_rows.append(
            {
                "name": f"Historical Player {number}",
                "position": "WR",
                "team": "BUF",
                "projected_score": 0.0,
                "actual_score": float(number),
            }
        )

    parquet_path = tmp_path / "season_2021_player_scores.parquet"
    write_processed_player_rows_to_parquet(processed_rows, parquet_path)

    result = run_historical_draft_simulation(
        parquet_path=parquet_path,
        team_count=4,
        roster_size=2,
        seed=2021,
        rebuild_if_missing=False,
    )

    summary = format_historical_simulation_summary(result)

    assert "Historical draft simulation complete" in summary
    assert "Final historical team scores:" in summary
    assert "Historical season winner:" in summary
    assert "Winning team draft picks:" in summary


def test_format_winning_team_draft_picks_shows_winning_roster(tmp_path):
    processed_rows = []

    for number in range(1, 21):
        processed_rows.append(
            {
                "name": f"Historical Player {number}",
                "position": "RB",
                "team": "ATL",
                "projected_score": 0.0,
                "actual_score": float(number),
            }
        )

    parquet_path = tmp_path / "season_2021_player_scores.parquet"
    write_processed_player_rows_to_parquet(processed_rows, parquet_path)

    result = run_historical_draft_simulation(
        parquet_path=parquet_path,
        team_count=4,
        roster_size=2,
        seed=2021,
        rebuild_if_missing=False,
    )

    winning_roster_text = format_winning_team_draft_picks(result)

    assert f"Winning team draft picks: {result.winner.team_name}" in winning_roster_text
    assert "Round" in winning_roster_text
    assert "Pick" in winning_roster_text
    assert "points" in winning_roster_text
