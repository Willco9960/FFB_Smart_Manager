"""Compact, device-resident scenario data for the CUDA migration path."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fantasy_engine.player import Player

POSITION_TO_ID = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
POSITION_NAMES = tuple(POSITION_TO_ID)


@dataclass(frozen=True)
class TensorScenarioBatch:
    """Static player data shared by batched draft/lineup kernels.

    All scenario tensors use the same player ordering. Keeping this contract
    explicit prevents repeated Python object conversion inside a generation.
    """

    projected_points: torch.Tensor
    actual_points: torch.Tensor
    positions: torch.Tensor
    player_keys: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.projected_points.ndim != 2 or self.actual_points.ndim != 2:
            raise ValueError("Point tensors must have shape [scenarios, players].")
        if self.projected_points.shape != self.actual_points.shape:
            raise ValueError("Projected and actual tensors must have the same shape.")
        if self.positions.shape != (self.projected_points.shape[1],):
            raise ValueError("positions must describe every player exactly once.")
        if len(self.player_keys) != self.projected_points.shape[1]:
            raise ValueError("player_keys must describe every player exactly once.")
        if self.projected_points.device != self.actual_points.device:
            raise ValueError("Point tensors must be on the same device.")

    @property
    def scenario_count(self) -> int:
        return self.projected_points.shape[0]

    @property
    def player_count(self) -> int:
        return self.projected_points.shape[1]

    def to(self, device: torch.device | str, non_blocking: bool = True) -> TensorScenarioBatch:
        target = torch.device(device)
        return TensorScenarioBatch(
            projected_points=self.projected_points.to(target, non_blocking=non_blocking),
            actual_points=self.actual_points.to(target, non_blocking=non_blocking),
            positions=self.positions.to(target, non_blocking=non_blocking),
            player_keys=self.player_keys,
        )

    def pin_memory(self) -> TensorScenarioBatch:
        return TensorScenarioBatch(
            projected_points=self.projected_points.pin_memory(),
            actual_points=self.actual_points.pin_memory(),
            positions=self.positions.pin_memory(),
            player_keys=self.player_keys,
        )

    @classmethod
    def from_player_scenarios(
        cls,
        scenarios: list[list[Player]],
        *,
        device: torch.device | str = "cpu",
    ) -> TensorScenarioBatch:
        if not scenarios or not scenarios[0]:
            raise ValueError("At least one non-empty player scenario is required.")
        first_keys = tuple((player.name, player.position) for player in scenarios[0])
        for scenario in scenarios[1:]:
            keys = tuple((player.name, player.position) for player in scenario)
            if keys != first_keys:
                raise ValueError("All scenarios must use identical player ordering.")
        try:
            position_ids = [POSITION_TO_ID[player.position] for player in scenarios[0]]
        except KeyError as error:
            raise ValueError(f"Unsupported position: {error.args[0]}") from error
        projected = torch.tensor(
            [[player.projected_score for player in scenario] for scenario in scenarios],
            dtype=torch.float32,
            device=device,
        )
        actual = torch.tensor(
            [[player.actual_score for player in scenario] for scenario in scenarios],
            dtype=torch.float32,
            device=device,
        )
        positions = torch.tensor(position_ids, dtype=torch.long, device=device)
        return cls(projected, actual, positions, first_keys)

    @classmethod
    def from_players(
        cls,
        players: list[Player],
        *,
        device: torch.device | str = "cpu",
    ) -> TensorScenarioBatch:
        return cls.from_player_scenarios([players], device=device)


def create_synthetic_scenario_batch(
    scenarios: int,
    players: int,
    *,
    device: torch.device | str = "cpu",
    seed: int = 20260815,
) -> TensorScenarioBatch:
    """Create reproducible benchmark data without Python-side per-pick work."""

    if scenarios < 1 or players < 1:
        raise ValueError("scenarios and players must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projected = torch.rand((scenarios, players), generator=generator) * 500.0
    actual = torch.rand((scenarios, players), generator=generator) * 500.0
    positions = torch.tensor(
        ([0, 1, 1, 2, 2, 3] * ((players + 5) // 6))[:players],
        dtype=torch.long,
    )
    player_keys = tuple(
        (f"player_{index}", POSITION_NAMES[position])
        for index, position in enumerate(positions.tolist())
    )
    return TensorScenarioBatch(
        projected_points=projected.to(device),
        actual_points=actual.to(device),
        positions=positions.to(device),
        player_keys=player_keys,
    )
