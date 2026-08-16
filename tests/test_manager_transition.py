import pytest

from fantasy_engine.manager_transition import (
    LegalActionMask,
    ManagerAction,
    ManagerState,
    ManagerTransition,
    build_manager_state,
)
from fantasy_engine.player import Player
from fantasy_engine.team import Team


def test_manager_transition_validates_shared_contract_and_mask():
    team = Team("Team 1", [Player("P1", "RB", "T", player_id="p1")])
    state = build_manager_state(team, [], season=2024, week=1)
    next_state = build_manager_state(team, [], season=2024, week=2)
    action = ManagerAction("lineup", "p1", "Team 1", ("p1",))
    mask = LegalActionMask("lineup", ("p1",), contract_digest=state.contract_digest)
    transition = ManagerTransition(state, action, mask, next_state, (("weekly_points", 10.0),))

    transition.validate()

    assert len(transition.digest()) == 64


def test_manager_transition_rejects_masked_action():
    state = ManagerState(2024, 1, "Team 1", (), ())
    action = ManagerAction("draft", "p1", "Team 1")
    mask = LegalActionMask("draft", ("p2",), contract_digest=state.contract_digest)
    transition = ManagerTransition(state, action, mask, state)

    with pytest.raises(ValueError, match="not allowed"):
        transition.validate()
