"""Offline decision replay records and weighted policy pretraining."""

from dataclasses import dataclass

import torch
from torch import nn

from models.modular_manager_policy import (
    DECISION_TYPES,
    ModularManagerPolicyNetwork,
    ModularPolicyFeatures,
)


@dataclass(frozen=True)
class DecisionReplayRecord:
    season: int
    week: int
    decision_type: str
    action_key: str
    features: ModularPolicyFeatures
    reward: float
    team_name: str = ""
    source: str = "historical"
    executed: bool = True
    executed: bool = True

    def __post_init__(self) -> None:
        if self.decision_type not in DECISION_TYPES:
            raise ValueError(f"Unknown decision type: {self.decision_type}")


class DecisionReplayBuffer:
    def __init__(self, records: list[DecisionReplayRecord] | None = None):
        self.records = list(records or [])

    def add(self, record: DecisionReplayRecord) -> None:
        self.records.append(record)

    def extend(self, records: list[DecisionReplayRecord]) -> None:
        self.records.extend(records)

    def by_decision_type(self, decision_type: str) -> list[DecisionReplayRecord]:
        return [record for record in self.records if record.decision_type == decision_type]

    def __len__(self) -> int:
        return len(self.records)


def train_offline_policy(
    model: ModularManagerPolicyNetwork,
    replay_buffer: DecisionReplayBuffer,
    epochs: int = 50,
    learning_rate: float = 0.001,
) -> float:
    """Fit policy heads to historical rewards without future-season access.

    Rewards are normalized per decision type.  Positive replay outcomes receive
    larger weights, but negative decisions remain in the data so the policy
    learns what to avoid rather than only copying winners.
    """

    if not replay_buffer.records:
        raise ValueError("At least one replay record is required.")

    reward_ranges = {}
    for decision_type in DECISION_TYPES:
        values = [
            record.reward
            for record in replay_buffer.records
            if record.decision_type == decision_type
        ]
        if values:
            reward_ranges[decision_type] = (min(values), max(values))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_function = nn.HuberLoss(reduction="none")
    grouped_records: dict[str, list[DecisionReplayRecord]] = {}
    for record in replay_buffer.records:
        grouped_records.setdefault(record.decision_type, []).append(record)
    final_loss = 0.0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        total_weight = 0.0
        for decision_type, decision_records in grouped_records.items():
            low, high = reward_ranges[decision_type]
            denominator = max(high - low, 1.0)
            players = torch.tensor(
                [record.features.player_values for record in decision_records],
                dtype=torch.float32,
            )
            states = torch.tensor(
                [record.features.state_values for record in decision_records],
                dtype=torch.float32,
            )
            targets = torch.tensor(
                [(record.reward - low) / denominator for record in decision_records],
                dtype=torch.float32,
            )
            weights = torch.tensor(
                [1.0 + min(abs(record.reward) / 100.0, 2.0) for record in decision_records],
                dtype=torch.float32,
            )
            predictions = model(players, states, decision_type=decision_type)
            total_loss = total_loss + (loss_function(predictions, targets) * weights).sum()
            total_weight += float(weights.sum().item())
        total_loss = total_loss / max(total_weight, 1.0)
        total_loss.backward()
        optimizer.step()
        final_loss = float(total_loss.item())

    return final_loss
