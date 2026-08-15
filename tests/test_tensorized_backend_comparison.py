import torch

from scripts.compare_tensorized_backends import create_inputs


def test_comparison_inputs_are_deterministic():
    args = type("Args", (), {"scenarios": 2, "players": 12})()
    first = create_inputs(args)
    second = create_inputs(args)

    for first_tensor, second_tensor in zip(first, second, strict=True):
        assert torch.equal(first_tensor, second_tensor)


def test_comparison_inputs_have_expected_shapes():
    args = type("Args", (), {"scenarios": 3, "players": 18})()
    projected, actual, positions = create_inputs(args)

    assert projected.shape == (3, 18)
    assert actual.shape == (3, 18)
    assert positions.shape == (18,)
