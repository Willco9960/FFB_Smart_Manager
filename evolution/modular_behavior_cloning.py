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

    losses = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_function = nn.HuberLoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        for example in examples:
            player = torch.tensor([example.features.player_values], dtype=torch.float32)
            state = torch.tensor([example.features.state_values], dtype=torch.float32)
            prediction = model(player, state, decision_type=example.decision_type)
            target = torch.tensor([example.target_score], dtype=torch.float32)
            total_loss = total_loss + loss_function(prediction, target)
        total_loss = total_loss / len(examples)
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.item()))

    return losses[-1]
