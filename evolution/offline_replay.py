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
    source: str = "historical"

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
    final_loss = 0.0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        total_weight = 0.0
        for record in replay_buffer.records:
            low, high = reward_ranges[record.decision_type]
            denominator = max(high - low, 1.0)
            target_value = (record.reward - low) / denominator
            player = torch.tensor([record.features.player_values], dtype=torch.float32)
            state = torch.tensor([record.features.state_values], dtype=torch.float32)
            prediction = model(
                player,
                state,
                decision_type=record.decision_type,
            )
            target = torch.tensor([target_value], dtype=torch.float32)
            weight = 1.0 + min(abs(record.reward) / 100.0, 2.0)
            total_loss = total_loss + (loss_function(prediction, target) * weight).sum()
            total_weight += weight
        total_loss = total_loss / max(total_weight, 1.0)
        total_loss.backward()
        optimizer.step()
        final_loss = float(total_loss.item())

    return final_loss
