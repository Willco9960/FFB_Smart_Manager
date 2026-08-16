import random

import torch

from evolution.opponent_archive import OpponentArchive


def test_opponent_archive_samples_and_updates_ratings():
    archive = OpponentArchive(max_size=3)
    archive.add(torch.nn.Linear(1, 1), label="a")
    archive.add(torch.nn.Linear(1, 1), label="b")
    archive.update_ratings([10.0, 1.0])
    assert archive.sample(4, random.Random(1))


def test_archive_updates_only_the_current_generation_entries():
    archive = OpponentArchive(max_size=8)
    first = archive.add(torch.nn.Linear(2, 1), label="first")
    second = archive.add(torch.nn.Linear(2, 1), label="second")

    archive.update_entry_ratings([first, second], [10.0, 0.0])

    assert first.rating > second.rating
    assert first.label == "first"
    assert second.label == "second"
