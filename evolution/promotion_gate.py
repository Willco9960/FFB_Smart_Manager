"""Unseen-season promotion gate with paired uncertainty estimates."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    mean_delta: float
    lower_delta: float
    upper_delta: float
    wins_delta: float
    reasons: tuple[str, ...]


def bootstrap_mean_interval(
    deltas: list[float],
    *,
    seed: int = 1,
    samples: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not deltas:
        raise ValueError("At least one paired delta is required.")
    if samples < 100:
        raise ValueError("At least 100 bootstrap samples are required.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one.")
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = int(tail * samples)
    upper_index = min(samples - 1, int((1.0 - tail) * samples))
    return means[lower_index], means[upper_index]


def evaluate_promotion_gate(
    candidate_fitness: list[float],
    baseline_fitness: list[float],
    candidate_wins: list[float],
    baseline_wins: list[float],
    *,
    seed: int = 1,
    minimum_mean_delta: float = 0.0,
    evaluation_seasons: tuple[int, ...] | list[int] | None = None,
    training_end_season: int | None = None,
) -> PromotionDecision:
    if not candidate_fitness or len(candidate_fitness) != len(baseline_fitness):
        raise ValueError("Candidate and baseline fitness must be aligned and non-empty.")
    if len(candidate_fitness) < 2:
        raise ValueError("Promotion gate requires at least two unseen seasons.")
    if len(candidate_wins) != len(baseline_wins) or len(candidate_wins) != len(candidate_fitness):
        raise ValueError("Win results must align with fitness results.")
    if evaluation_seasons is not None:
        if len(evaluation_seasons) != len(candidate_fitness):
            raise ValueError("Evaluation seasons must align with fitness results.")
        if len(set(evaluation_seasons)) != len(evaluation_seasons):
            raise ValueError("Evaluation seasons must be unique.")
        if training_end_season is not None and any(
            season <= training_end_season for season in evaluation_seasons
        ):
            raise ValueError("Promotion evaluation seasons must be after training.")
    deltas = [
        candidate - baseline
        for candidate, baseline in zip(candidate_fitness, baseline_fitness, strict=True)
    ]
    wins_delta = sum(candidate_wins) / len(candidate_wins) - sum(baseline_wins) / len(baseline_wins)
    lower, upper = bootstrap_mean_interval(deltas, seed=seed)
    mean_delta = sum(deltas) / len(deltas)
    reasons = []
    if mean_delta < minimum_mean_delta:
        reasons.append("mean fitness did not improve over baseline")
    if lower <= 0.0:
        reasons.append("paired bootstrap interval crosses zero")
    if wins_delta < 0.0:
        reasons.append("weekly wins regressed")
    return PromotionDecision(
        promoted=not reasons,
        mean_delta=round(mean_delta, 4),
        lower_delta=round(lower, 4),
        upper_delta=round(upper, 4),
        wins_delta=round(wins_delta, 4),
        reasons=tuple(reasons),
    )
