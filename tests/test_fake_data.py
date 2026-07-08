from fantasy_engine.fake_data import POSITION_COUNTS, create_fake_player_pool


def test_fake_player_pool_has_enough_players_for_10_team_16_round_draft():
    players = create_fake_player_pool()

    assert len(players) >= 160


def test_fake_player_pool_has_required_positions():
    players = create_fake_player_pool()
    positions = {player.position for player in players}

    assert "QB" in positions
    assert "RB" in positions
    assert "WR" in positions
    assert "TE" in positions
    assert "K" in positions
    assert "DST" in positions


def test_fake_player_pool_position_counts_match_config():
    players = create_fake_player_pool()

    for position, expected_count in POSITION_COUNTS.items():
        actual_count = len([player for player in players if player.position == position])

        assert actual_count == expected_count


def test_fake_player_pool_assigns_projected_and_actual_scores():
    players = create_fake_player_pool()

    for player in players:
        assert isinstance(player.projected_score, float)
        assert isinstance(player.actual_score, float)
