"""Stable score utilities shared by the neural decision agents.

Policy heads are trained on different targets and can have very different
numeric scales.  Decisions should therefore combine them by rank/percentile,
not by trusting an uncalibrated raw output.  The helpers here also provide a
small transparent football prior so an early or partially trained head cannot
select a clearly inferior player just because its output scale drifted.
"""

from __future__ import annotations

import math


def normalize_scores(scores: list[float]) -> list[float]:
    """Map a candidate score list to ``[0, 1]`` without changing ordering."""

    if not scores:
        return []

    low = min(scores)
    high = max(scores)
    spread = high - low
    if spread <= 1e-9:
        return [0.5 for _ in scores]

    return [(score - low) / spread for score in scores]


def blend_policy_and_anchor_scores(
    policy_scores: list[float],
    anchor_scores: list[float],
    policy_weight: float = 0.80,
) -> list[float]:
    """Blend normalized learned scores with a transparent prior.

    The anchor is deliberately a minority vote.  It protects the system from
    an uncalibrated head while allowing evolution to override it when the
    learned policy finds a better pattern.
    """

    if len(policy_scores) != len(anchor_scores):
        raise ValueError("Policy and anchor score lists must have equal length.")
    if not 0.0 <= policy_weight <= 1.0:
        raise ValueError("policy_weight must be between zero and one.")

    normalized_policy = normalize_scores(policy_scores)
    normalized_anchor = normalize_scores(anchor_scores)
    anchor_weight = 1.0 - policy_weight
    return [
        (policy_weight * policy) + (anchor_weight * anchor)
        for policy, anchor in zip(normalized_policy, normalized_anchor, strict=True)
    ]


def bounded_policy_score(score: float) -> float:
    """Bound a single policy output for additive transaction scoring."""

    bounded = 0.5 + (0.5 * math.tanh(score))
    return min(max(bounded, 1e-6), 1.0 - 1e-6)
