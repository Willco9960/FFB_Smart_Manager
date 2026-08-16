"""Experimental tensorized simulation backends.

The production object-oriented simulator remains in ``fantasy_engine``.
Modules in this package are intentionally benchmark-only until their outputs
match the production reference implementation.
"""

from gpu_sim.full_season import (
    CudaSeasonState,
    create_synthetic_season_state,
    run_full_cuda_season,
)
from gpu_sim.historical_adapter import (
    HistoricalCudaInputs,
    create_historical_cuda_inputs,
)
from gpu_sim.profiling import TensorStageProfiler
from gpu_sim.policy_training import (
    CudaGenerationMetrics,
    CudaPolicyEvaluation,
    evaluate_cuda_policy,
    save_cuda_policy_checkpoint,
    train_cuda_policy_population,
)
from gpu_sim.tensor_state import TensorScenarioBatch, create_synthetic_scenario_batch
from gpu_sim.tensorized_draft import (
    DraftBatchResult,
    LineupBatchResult,
    benchmark_tensorized_draft,
    benchmark_tensorized_for_duration,
    run_batched_greedy_draft,
    run_batched_roster_aware_draft,
    score_batched_lineups,
    score_batched_offensive_lineups,
)

__all__ = [
    "DraftBatchResult",
    "LineupBatchResult",
    "benchmark_tensorized_draft",
    "benchmark_tensorized_for_duration",
    "run_batched_greedy_draft",
    "run_batched_roster_aware_draft",
    "score_batched_offensive_lineups",
    "score_batched_lineups",
    "TensorScenarioBatch",
    "create_synthetic_scenario_batch",
    "TensorStageProfiler",
    "CudaGenerationMetrics",
    "CudaPolicyEvaluation",
    "evaluate_cuda_policy",
    "save_cuda_policy_checkpoint",
    "train_cuda_policy_population",
    "CudaSeasonState",
    "create_synthetic_season_state",
    "run_full_cuda_season",
    "HistoricalCudaInputs",
    "create_historical_cuda_inputs",
]
