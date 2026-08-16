"""Frozen opponent archive and lightweight Elo bookkeeping for self-play."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class ArchivedOpponent:
    policy: torch.nn.Module
    rating: float = 1500.0
    label: str = ""


class OpponentArchive:
    def __init__(self, max_size: int = 64, initial_rating: float = 1500.0):
        if max_size < 1:
            raise ValueError("max_size must be positive.")
        self.max_size = max_size
        self.initial_rating = initial_rating
        self.entries: list[ArchivedOpponent] = []

    def add(self, policy: torch.nn.Module, label: str = "") -> ArchivedOpponent:
        entry = ArchivedOpponent(policy=policy, rating=self.initial_rating, label=label)
        self.entries.append(entry)
        if len(self.entries) > self.max_size:
            self.entries.sort(key=lambda entry: entry.rating, reverse=True)
            del self.entries[self.max_size :]
        return entry

    def sample(self, count: int, rng: random.Random) -> list[torch.nn.Module]:
        if count < 1 or not self.entries:
            return []
        ranked = sorted(self.entries, key=lambda entry: entry.rating, reverse=True)
        # Keep strong opponents common while retaining exploration of the archive.
        minimum_rating = min(item.rating for item in ranked)
        weights = [max(1.0, entry.rating - minimum_rating + 1.0) for entry in ranked]
        selected = rng.choices(ranked, weights=weights, k=count)
        return [entry.policy for entry in selected]

    def update_ratings(self, fitnesses: list[float], k_factor: float = 16.0) -> None:
        self.update_entry_ratings(self.entries, fitnesses, k_factor=k_factor)

    def update_entry_ratings(
        self,
        entries: list[ArchivedOpponent],
        fitnesses: list[float],
        k_factor: float = 16.0,
    ) -> None:
        """Update only the entries evaluated in the current generation.

        The archive contains policies from many generations, while a training
        generation evaluates only the current population.  Updating a prefix
        or assigning the same score to archived policies would corrupt the
        difficulty sampler, so ratings are updated against the selected
        generation only and then retained in the archive.
        """
        if len(fitnesses) != len(entries):
            raise ValueError("Fitness count must match selected archive entries.")
        if not entries:
            return

        for index, entry in enumerate(entries):
            expected = sum(
                1.0 / (1.0 + 10.0 ** ((other.rating - entry.rating) / 400.0))
                for other in entries
                if other is not entry
            ) / max(len(entries) - 1, 1)
            actual = sum(
                fitnesses[index] > other_fitness
                for other_index, other_fitness in enumerate(fitnesses)
                if other_index != index
            ) / max(len(fitnesses) - 1, 1)
            entry.rating += k_factor * (actual - expected)
