"""Train the transaction value model from historical downstream rewards."""

from statistics import mean, pstdev

import torch

from evolution.offline_replay import DecisionReplayRecord
from models.transaction_value import TRANSACTION_DECISION_TYPES, TransactionValueNetwork


def train_transaction_value_model(
    model: TransactionValueNetwork,
    records: list[DecisionReplayRecord],
    epochs: int = 50,
    learning_rate: float = 0.001,
) -> tuple[float, int]:
    """Fit waiver/trade value and uncertainty to future transaction rewards."""

    transaction_records = {
        decision_type: [record for record in records if record.decision_type == decision_type]
        for decision_type in TRANSACTION_DECISION_TYPES
    }
    if not any(transaction_records.values()):
        raise ValueError("At least one waiver or trade replay record is required.")

    for index, decision_type in enumerate(TRANSACTION_DECISION_TYPES):
        rewards = [record.reward for record in transaction_records[decision_type]]
        if not rewards:
            continue
        model.target_means[index] = mean(rewards)
        model.target_scales[index] = max(pstdev(rewards), 1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    final_loss = 0.0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        total_count = 0
        for index, decision_type in enumerate(TRANSACTION_DECISION_TYPES):
            decision_records = transaction_records[decision_type]
            if not decision_records:
                continue
            players = torch.tensor(
                [record.features.player_values for record in decision_records],
                dtype=torch.float32,
            )
            states = torch.tensor(
                [
                    model._fit_state_features(record.features.state_values)
                    for record in decision_records
                ],
                dtype=torch.float32,
            )
            targets = torch.tensor(
                [
                    (record.reward - float(model.target_means[index].item()))
                    / float(model.target_scales[index].item())
                    for record in decision_records
                ],
                dtype=torch.float32,
            )
            raw_output = model(players, states, decision_type)
            predicted_mean = raw_output[:, 0]
            predicted_scale = torch.nn.functional.softplus(raw_output[:, 1]) + 0.05
            nll = 0.5 * ((targets - predicted_mean) / predicted_scale) ** 2
            nll = nll + torch.log(predicted_scale)
            total_loss = total_loss + nll.mean()
            total_count += 1

        if total_count == 0:
            raise ValueError("At least one waiver or trade replay record is required.")
        total_loss = total_loss / total_count
        total_loss.backward()
        optimizer.step()
        final_loss = float(total_loss.item())

    return final_loss, sum(len(items) for items in transaction_records.values())
