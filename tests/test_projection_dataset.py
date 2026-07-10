from fantasy_engine.projection_dataset import (
    create_projection_examples,
    get_feature_names,
    get_position_features,
)


def create_season_player(
    name: str, position: str, team: str, points: float
) -> dict[str, str | float]:
    return {
        "name": name,
        "position": position,
        "team": team,
        "fantasy_points": points,
    }


def test_get_feature_names_matches_feature_count():
    assert len(get_feature_names()) == 7


def test_get_position_features_uses_one_hot_encoding():
    assert get_position_features("WR") == (0.0, 0.0, 1.0, 0.0)


def test_create_projection_examples_uses_prior_season_only():
    season_totals = {
        2020: {("Test RB", "RB"): create_season_player("Test RB", "RB", "ATL", 100.0)},
        2021: {("Test RB", "RB"): create_season_player("Test RB", "RB", "BUF", 150.0)},
    }

    examples = create_projection_examples(season_totals, [2021])

    assert len(examples) == 1
    assert examples[0].features[0] == 100.0
    assert examples[0].features[1] == 100.0
    assert examples[0].features[2] == 1.0
    assert examples[0].target_points == 150.0


def test_create_projection_examples_excludes_players_without_prior_season_data():
    season_totals = {
        2020: {},
        2021: {("Rookie WR", "WR"): create_season_player("Rookie WR", "WR", "ATL", 150.0)},
    }

    assert create_projection_examples(season_totals, [2021]) == []
