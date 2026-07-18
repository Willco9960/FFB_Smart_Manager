"""Behavioral pretraining for the modular policy.

This is the warm-start stage before self-play.  It teaches the draft head to
imitate a transparent projection/scarcity teacher so evolution does not start
from random decisions.
"""

from dataclasses import dataclass

import torch
from torch import nn

from models.modular_manager_policy import ModularManagerPolicyNetwork, ModularPolicyFeatures


@dataclass(frozen=True)
class ModularImitationExample:
    features: ModularPolicyFeatures
    target_score: float
    decision_type: str = "draft"


def train_modular_behavior_policy(
    model: ModularManagerPolicyNetwork,
    examples: list[ModularImitationExample],
    epochs: int = 100,
    learning_rate: float = 0.003,
) -> float:
    if not examples:
        raise ValueError("At least one imitation example is required.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_function = nn.HuberLoss()
    grouped_examples: dict[str, list[ModularImitationExample]] = {}
    for example in examples:
        grouped_examples.setdefault(example.decision_type, []).append(example)
    final_loss = 0.0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        for decision_type, decision_examples in grouped_examples.items():
            players = torch.tensor(
                [example.features.player_values for example in decision_examples],
                dtype=torch.float32,
            )
            states = torch.tensor(
                [example.features.state_values for example in decision_examples],
                dtype=torch.float32,
            )
            targets = torch.tensor(
                [example.target_score for example in decision_examples],
                dtype=torch.float32,
            )
            predictions = model(players, states, decision_type=decision_type)
            total_loss = total_loss + loss_function(predictions, targets).mean()
        total_loss = total_loss / len(grouped_examples)
        total_loss.backward()
        optimizer.step()

        final_loss = float(total_loss.item())

    return final_loss
