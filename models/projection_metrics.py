"""Decision-focused projection metrics used by validation and promotion gates."""

from __future__ import annotations


def mean_absolute_error(predictions: list[float], targets: list[float]) -> float:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Predictions and targets must be non-empty and aligned.")
    return sum(
        abs(prediction - target)
        for prediction, target in zip(predictions, targets, strict=True)
    ) / len(targets)


def spearman_rank_correlation(predictions: list[float], targets: list[float]) -> float:
    if len(predictions) != len(targets) or len(predictions) < 2:
        raise ValueError("At least two aligned predictions and targets are required.")

    def ranks(values: list[float]) -> list[float]:
        ordered = sorted(enumerate(values), key=lambda item: item[1])
        result = [0.0] * len(values)
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][1] == ordered[index][1]:
                end += 1
            rank = (index + end - 1) / 2.0 + 1.0
            for position in range(index, end):
                result[ordered[position][0]] = rank
            index = end
        return result

    predicted_ranks = ranks(predictions)
    target_ranks = ranks(targets)
    predicted_mean = sum(predicted_ranks) / len(predicted_ranks)
    target_mean = sum(target_ranks) / len(target_ranks)
    numerator = sum(
        (left - predicted_mean) * (right - target_mean)
        for left, right in zip(predicted_ranks, target_ranks, strict=True)
    )
    left_norm = sum((value - predicted_mean) ** 2 for value in predicted_ranks) ** 0.5
    right_norm = sum((value - target_mean) ** 2 for value in target_ranks) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return round(numerator / (left_norm * right_norm), 12)


def top_k_hit_rate(
    predictions: list[float],
    targets: list[float],
    k: int,
) -> float:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Predictions and targets must be non-empty and aligned.")
    if k < 1:
        raise ValueError("k must be positive.")
    k = min(k, len(predictions))
    predicted_top = {
        index
        for index, _ in sorted(enumerate(predictions), key=lambda item: item[1], reverse=True)[:k]
    }
    actual_top = {
        index for index, _ in sorted(enumerate(targets), key=lambda item: item[1], reverse=True)[:k]
    }
    return len(predicted_top & actual_top) / k


def quantile_coverage(
    lower: list[float],
    upper: list[float],
    targets: list[float],
) -> float:
    if not (len(lower) == len(upper) == len(targets)) or not targets:
        raise ValueError("Quantile bounds and targets must be non-empty and aligned.")
    return sum(
        low <= target <= high for low, high, target in zip(lower, upper, targets, strict=True)
    ) / len(targets)


def calibration_error(probabilities: list[float], outcomes: list[bool], bins: int = 10) -> float:
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("Probabilities and outcomes must be non-empty and aligned.")
    if bins < 1:
        raise ValueError("bins must be positive.")
    error = 0.0
    total = len(probabilities)
    for bin_index in range(bins):
        members = [
            (probability, outcome)
            for probability, outcome in zip(probabilities, outcomes, strict=True)
            if min(bin_index / bins, 0.999999) <= probability < (bin_index + 1) / bins
        ]
        if not members:
            continue
        predicted = sum(probability for probability, _ in members) / len(members)
        observed = sum(outcome for _, outcome in members) / len(members)
        error += (len(members) / total) * abs(predicted - observed)
    return error


def lineup_regret(
    predicted_scores: list[float], actual_scores: list[float], lineup_size: int
) -> float:
    if len(predicted_scores) != len(actual_scores) or not predicted_scores:
        raise ValueError("Predicted and actual scores must be non-empty and aligned.")
    if lineup_size < 1:
        raise ValueError("lineup_size must be positive.")
    lineup_size = min(lineup_size, len(predicted_scores))
    chosen = sorted(range(len(predicted_scores)), key=predicted_scores.__getitem__, reverse=True)[
        :lineup_size
    ]
    optimal = sorted(actual_scores, reverse=True)[:lineup_size]
    return max(0.0, sum(optimal) - sum(actual_scores[index] for index in chosen))
