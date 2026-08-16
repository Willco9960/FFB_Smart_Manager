"""Fast, fail-closed checks required before expensive manager training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from evolution.pretraining import build_manager_teacher_examples, run_manager_pretraining
from fantasy_engine.data_availability import DataAvailabilityManifest, validate_training_seasons
from fantasy_engine.fitness_contract import ESPN_FITNESS_CONTRACT
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from gpu_sim.full_season import create_synthetic_season_state, run_full_cuda_season
from models.modular_manager_policy import ModularManagerPolicyNetwork


@dataclass(frozen=True)
class PreflightResult:
    data_ready: bool
    policy_heads_ready: bool
    contract_ready: bool
    manifests: tuple[DataAvailabilityManifest, ...]
    pretraining_loss: float

    @property
    def approved(self) -> bool:
        return self.data_ready and self.policy_heads_ready and self.contract_ready

    def to_dict(self) -> dict[str, object]:
        return {
            "approved": self.approved,
            "data_ready": self.data_ready,
            "policy_heads_ready": self.policy_heads_ready,
            "contract_ready": self.contract_ready,
            "pretraining_loss": self.pretraining_loss,
            "manifests": [manifest.to_dict() for manifest in self.manifests],
        }


def run_training_preflight(
    seasons: tuple[int, ...],
    *,
    device: torch.device | str = "cpu",
) -> PreflightResult:
    if not seasons:
        raise ValueError("At least one training season is required.")
    availability_seasons = tuple(
        sorted({value for season in seasons for value in (season - 1, season)})
    )
    manifests = validate_training_seasons(availability_seasons)
    players = [
        Player(f"Preflight {index}", position, "T", projected_score=20.0 - index)
        for index, position in enumerate(("QB", "RB", "WR", "TE", "K", "DST") * 4)
    ]
    league = League(
        "Preflight league",
        [Team(f"Team {index}") for index in range(1, 11)],
        players,
    )
    model = ModularManagerPolicyNetwork()
    examples = build_manager_teacher_examples(league, rounds=2)
    pretraining = run_manager_pretraining(
        model,
        examples,
        behavior_epochs=1,
        device=device,
    )
    state = create_synthetic_season_state(
        scenarios=1,
        players=200,
        weeks=17,
        device=device,
    )
    run_full_cuda_season(
        state,
        policy_network=model,
        policy_team_indices=torch.zeros(1, dtype=torch.long, device=state.device),
        enable_transactions=False,
    )
    return PreflightResult(
        data_ready=True,
        policy_heads_ready=pretraining.approved,
        contract_ready=state.contract_digest == ESPN_FITNESS_CONTRACT.digest(),
        manifests=manifests,
        pretraining_loss=pretraining.behavior_loss,
    )
