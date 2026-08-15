"""Island-model orchestration for modular manager-policy evolution.

Each island evolves independently for a bounded segment, then exchanges one
elite policy with a neighboring island.  This preserves evolutionary
dependency within an island while allowing independent segments to use CPU
cores concurrently.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import torch

from evolution.genome import DraftStrategyGenome
from evolution.modular_policy_training import (
    clone_modular_policy,
    train_modular_policy_self_play,
)
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from models.modular_manager_policy import ModularManagerPolicyNetwork


@dataclass(frozen=True)
class IslandSegmentResult:
    island_index: int
    segment_number: int
    best_score: float
    policy: ModularManagerPolicyNetwork
    history: list[float]


@dataclass(frozen=True)
class IslandTrainingResult:
    best_policy: ModularManagerPolicyNetwork
    best_score: float
    segment_scores: list[list[float]]
    island_scores: list[list[float]]


def _initialize_island_worker() -> None:
    """Keep one island process from consuming every BLAS thread."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _run_island_segment(payload) -> IslandSegmentResult:
    (
        island_index,
        segment_number,
        initial_policy,
        scenarios,
        transaction_genome,
        population_size,
        generations_per_segment,
        selection_count,
        mutation_strength,
        final_mutation_strength,
        seed,
        rounds,
        lineup_rules,
        scenarios_per_generation,
        full_evaluation_interval,
        anchor_scenarios_per_generation,
        risk_penalty,
    ) = payload
    trained_policy, history = train_modular_policy_self_play(
        initial_policy=initial_policy,
        scenarios=scenarios,
        transaction_genome=transaction_genome,
        population_size=population_size,
        generations=generations_per_segment,
        selection_count=selection_count,
        mutation_strength=mutation_strength,
        final_mutation_strength=final_mutation_strength,
        seed=seed,
        rounds=rounds,
        lineup_rules=lineup_rules,
        scenarios_per_generation=scenarios_per_generation,
        full_evaluation_interval=full_evaluation_interval,
        anchor_scenarios_per_generation=anchor_scenarios_per_generation,
        risk_penalty=risk_penalty,
        evaluation_workers=1,
        run_final_evaluation=False,
    )
    return IslandSegmentResult(
        island_index=island_index,
        segment_number=segment_number,
        best_score=max(history) if history else float("-inf"),
        policy=trained_policy,
        history=list(history),
    )


def train_island_policy_self_play(
    initial_policy: ModularManagerPolicyNetwork,
    scenarios: list[tuple[League, list[WeeklyPlayerPerformance]]],
    transaction_genome: DraftStrategyGenome,
    island_count: int = 10,
    segments: int = 10,
    generations_per_segment: int = 10,
    population_size: int = 24,
    selection_count: int = 8,
    mutation_strength: float = 0.01,
    final_mutation_strength: float | None = None,
    seed: int = 1,
    rounds: int = 16,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    scenarios_per_generation: int | None = 8,
    full_evaluation_interval: int = 5,
    anchor_scenarios_per_generation: int = 4,
    risk_penalty: float = 0.10,
    island_workers: int | None = None,
) -> IslandTrainingResult:
    """Train independent islands with ring migration at segment barriers."""

    if island_count < 1:
        raise ValueError("island_count must be at least one.")
    if segments < 1 or generations_per_segment < 1:
        raise ValueError("segments and generations_per_segment must be at least one.")
    if population_size < selection_count or selection_count < 1:
        raise ValueError("selection_count must be between one and population_size.")
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    if island_workers is None:
        island_workers = island_count
    if island_workers < 1:
        raise ValueError("island_workers must be at least one.")

    policies = [clone_modular_policy(initial_policy) for _ in range(island_count)]
    global_best_policy = clone_modular_policy(initial_policy)
    global_best_score = float("-inf")
    segment_scores: list[list[float]] = []
    island_scores: list[list[float]] = [[] for _ in range(island_count)]

    for segment_number in range(1, segments + 1):
        payloads = [
            (
                island_index,
                segment_number,
                clone_modular_policy(policy),
                scenarios,
                transaction_genome,
                population_size,
                generations_per_segment,
                selection_count,
                mutation_strength,
                final_mutation_strength,
                seed + (segment_number * 10_000) + island_index,
                rounds,
                lineup_rules,
                scenarios_per_generation,
                full_evaluation_interval,
                anchor_scenarios_per_generation,
                risk_penalty,
            )
            for island_index, policy in enumerate(policies)
        ]
        worker_count = min(island_workers, island_count)
        if worker_count == 1:
            results = [_run_island_segment(payload) for payload in payloads]
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_initialize_island_worker,
            ) as executor:
                results = list(executor.map(_run_island_segment, payloads))
        results.sort(key=lambda result: result.island_index)

        scores = [result.best_score for result in results]
        segment_scores.append(scores)
        for result in results:
            island_scores[result.island_index].append(result.best_score)
            if result.best_score > global_best_score:
                global_best_score = result.best_score
                global_best_policy = clone_modular_policy(result.policy)

        # Ring migration keeps islands different while still sharing progress.
        # Island 0 receives the global elite; every other island receives the
        # previous island's elite from this segment.
        migrated_policies = [clone_modular_policy(global_best_policy)]
        migrated_policies.extend(
            clone_modular_policy(results[index - 1].policy) for index in range(1, island_count)
        )
        policies = migrated_policies

    return IslandTrainingResult(
        best_policy=global_best_policy,
        best_score=global_best_score,
        segment_scores=segment_scores,
        island_scores=island_scores,
    )
