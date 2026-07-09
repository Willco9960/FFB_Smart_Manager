import pytest

from fantasy_engine.lineup import (
    ESPN_DEFAULT_LINEUP_RULES,
    ESPN_OFFENSIVE_LINEUP_RULES,
    build_best_starting_lineup,
    score_starting_lineup,
)
from fantasy_engine.player import Player


def create_player(
    name: str,
    position: str,
    actual_score: float,
) -> Player:
    return Player(
        name=name,
        position=position,
        team="TEST",
        projected_score=actual_score,
        actual_score=actual_score,
    )


def test_build_best_starting_lineup_uses_espn_default_slots():
    roster = [
        create_player("QB 1", "QB", 300.0),
        create_player("RB 1", "RB", 200.0),
        create_player("RB 2", "RB", 190.0),
        create_player("RB 3", "RB", 180.0),
        create_player("WR 1", "WR", 210.0),
        create_player("WR 2", "WR", 205.0),
        create_player("TE 1", "TE", 120.0),
        create_player("DST 1", "DST", 100.0),
        create_player("K 1", "K", 95.0),
    ]

    starting_lineup = build_best_starting_lineup(
        roster=roster,
        lineup_rules=ESPN_DEFAULT_LINEUP_RULES,
    )

    assert starting_lineup.is_complete()
    assert len(starting_lineup.players) == 9


def test_score_starting_lineup_ignores_bench_players():
    roster = [
        create_player("QB 1", "QB", 300.0),
        create_player("QB Bench", "QB", 299.0),
        create_player("RB 1", "RB", 200.0),
        create_player("RB 2", "RB", 190.0),
        create_player("RB 3", "RB", 180.0),
        create_player("WR 1", "WR", 210.0),
        create_player("WR 2", "WR", 205.0),
        create_player("TE 1", "TE", 120.0),
        create_player("DST 1", "DST", 100.0),
        create_player("K 1", "K", 95.0),
    ]

    score = score_starting_lineup(
        roster=roster,
        lineup_rules=ESPN_DEFAULT_LINEUP_RULES,
    )

    assert score == 1600.0


def test_offensive_lineup_uses_qb_rb_wr_te_and_flex():
    roster = [
        create_player("QB 1", "QB", 300.0),
        create_player("RB 1", "RB", 200.0),
        create_player("RB 2", "RB", 190.0),
        create_player("RB Flex", "RB", 180.0),
        create_player("WR 1", "WR", 210.0),
        create_player("WR 2", "WR", 205.0),
        create_player("TE 1", "TE", 120.0),
    ]

    score = score_starting_lineup(
        roster=roster,
        lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
    )

    assert score == 1405.0


def test_build_best_starting_lineup_raises_when_required_slot_missing():
    roster = [
        create_player("QB 1", "QB", 300.0),
    ]

    with pytest.raises(ValueError):
        build_best_starting_lineup(
            roster=roster,
            lineup_rules=ESPN_DEFAULT_LINEUP_RULES,
        )


def test_build_best_starting_lineup_can_allow_incomplete_lineup():
    roster = [
        create_player("QB 1", "QB", 300.0),
    ]

    starting_lineup = build_best_starting_lineup(
        roster=roster,
        lineup_rules=ESPN_DEFAULT_LINEUP_RULES,
        require_complete_lineup=False,
    )

    assert not starting_lineup.is_complete()
    assert starting_lineup.score() == 300.0
