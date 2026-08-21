import pytest
import torch

from fantasy_engine.transaction_contract import (
    TransactionEvent,
    canonical_trade_action_key,
    canonical_transaction_state_digest,
    canonical_waiver_action_key,
    stable_argmax,
    stable_argmin,
    stable_best,
    stable_topk,
)


def test_cpu_and_cuda_contract_choose_same_first_tied_action():
    scores = [10.0, 10.0, 9.0]
    cpu_choice = stable_best(
        list(enumerate(scores)),
        score=lambda item: item[1],
        tie_key=lambda item: item[0],
    )[0]
    cuda_choice = stable_argmax(torch.tensor(scores), tie_break_indices=torch.arange(3))

    assert cpu_choice == cuda_choice.item() == 0


def test_contract_uses_explicit_action_key_not_input_order():
    actions = [
        {"action": "later", "score": 5.0, "key": 2},
        {"action": "earlier", "score": 5.0, "key": 1},
    ]

    selected = stable_best(
        actions,
        score=lambda item: item["score"],
        tie_key=lambda item: item["key"],
    )

    assert selected["action"] == "earlier"




def test_canonical_transaction_action_keys_are_backend_independent():
    assert canonical_waiver_action_key("Team 1", "p2|RB|A", "p1|RB|A") == (
        "waiver|Team 1|p2|RB|A|p1|RB|A"
    )
    assert canonical_trade_action_key(
        "Team 2", "Team 1", ["p3", "p2"], ["p4"]
    ) == "trade|Team 2|Team 1|p2,p3|p4"


def test_transaction_event_requires_rejection_reason_and_has_stable_digest():
    event = TransactionEvent(
        season=2023,
        week=4,
        sequence_index=2,
        decision_type="waiver",
        team_name="Team 1",
        action_key="waiver|Team 1|p2|p1",
        pre_state_digest="before",
        post_state_digest="after",
        reward_components=(("realized_week_gain", 1.5),),
    )
    assert event.digest() == event.digest()
    with pytest.raises(ValueError, match="rejection_reason"):
        TransactionEvent(
            season=2023,
            week=4,
            sequence_index=2,
            decision_type="trade",
            team_name="Team 1",
            action_key="trade|invalid",
            pre_state_digest="before",
            post_state_digest="after",
            accepted=False,
        )


def test_transaction_state_digest_uses_published_score_precision():
    base = {
        "team_rosters": (("team-0", ("p0",)),),
        "available_player_keys": ("p1",),
        "standings": (("team-0", 1, 75.04),),
    }

    assert canonical_transaction_state_digest(**base) == canonical_transaction_state_digest(
        **{**base, "standings": (("team-0", 1, 75.040009),)}
    )


def test_cuda_contract_stable_min_and_topk_match_cpu_order():
    scores = torch.tensor([3.0, 3.0, 2.0, 3.0])
    tie_keys = torch.tensor([30, 10, 20, 0])

    assert stable_argmin(scores, tie_break_indices=tie_keys).item() == 2
    assert stable_topk(scores, 3, tie_break_indices=tie_keys).tolist() == [3, 1, 0]
