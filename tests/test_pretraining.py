from evolution.pretraining import build_manager_teacher_examples, run_manager_pretraining
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from models.modular_manager_policy import DECISION_TYPES, ModularManagerPolicyNetwork


def test_teacher_examples_cover_all_decision_heads():
    players = [
        Player(f"P{index}", position, "T", projected_score=20.0 - index)
        for index, position in enumerate(("QB", "RB", "WR", "TE", "K", "DST") * 3)
    ]
    league = League("teacher", [Team(f"Team {index}") for index in range(1, 11)], players)

    examples = build_manager_teacher_examples(league, rounds=2)
    counts = {decision_type: 0 for decision_type in DECISION_TYPES}
    for example in examples:
        counts[example.decision_type] += 1

    assert all(counts.values())


def test_manager_pretraining_gate_returns_finite_result():
    players = [
        Player(f"P{index}", "RB", "T", projected_score=20.0 - index)
        for index in range(12)
    ]
    league = League("teacher", [Team(f"Team {index}") for index in range(1, 11)], players)
    examples = build_manager_teacher_examples(league, rounds=2)
    model = ModularManagerPolicyNetwork()

    result = run_manager_pretraining(model, examples, behavior_epochs=1)

    assert result.approved
    assert result.example_count == len(examples)
