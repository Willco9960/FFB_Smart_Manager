from fantasy_engine.weekly_projection_dataset import (
    WEEKLY_PROJECTION_FEATURE_NAMES,
    create_defensive_weekly_signals,
    create_weekly_projection_examples,
    get_opponent_signal_averages,
)


def row(**values: str) -> dict[str, str]:
    defaults = {
        "player_name": "Player",
        "position": "WR",
        "recent_team": "OFF",
        "opponent_team": "DEF",
        "week": "1",
        "receiving_yards": "80",
        "receptions": "5",
        "targets": "8",
    }
    defaults.update(values)
    return defaults


def test_weekly_features_have_documented_feature_count():
    assert len(WEEKLY_PROJECTION_FEATURE_NAMES) == 27


def test_opponent_defensive_signal_uses_only_prior_weeks():
    rows = [
        row(position="DE", recent_team="DEF", week="1", def_sacks="2"),
        row(position="DE", recent_team="DEF", week="3", def_sacks="8"),
    ]
    signals = create_defensive_weekly_signals(rows)

    assert get_opponent_signal_averages(signals, "DEF", 3)[0] == 2.0


def test_weekly_examples_do_not_include_current_week_in_recent_features():
    prior_rows = [row(week="1", receiving_yards="60")]
    target_rows = [row(week="2", receiving_yards="200")]
    older_rows = [row(week="1", receiving_yards="40")]

    examples = create_weekly_projection_examples(
        season_rows_by_season={2020: older_rows, 2021: prior_rows + target_rows},
        target_seasons=[2021],
    )

    target_example = next(example for example in examples if example.week == 2)

    assert target_example.features[2] < target_example.target_points
