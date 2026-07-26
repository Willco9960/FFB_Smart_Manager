"""Train and validate the transaction value model from historical rewards."""

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

import torch

from evolution.offline_replay import DecisionReplayRecord
from models.transaction_value import TRANSACTION_DECISION_TYPES, TransactionValueNetwork


@dataclass(frozen=True)
class TransactionValueValidation:
    """Chronological holdout metrics used to decide whether a model is safe."""

    train_records: int
    validation_records: int
    mae: float | None
    baseline_mae: float | None
    sign_accuracy: float | None
    uncertainty_coverage: float | None
    approved: bool
    holdout_seasons: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_records": self.train_records,
            "validation_records": self.validation_records,
            "mae": self.mae,
            "baseline_mae": self.baseline_mae,
            "sign_accuracy": self.sign_accuracy,
            "uncertainty_coverage": self.uncertainty_coverage,
            "approved": self.approved,
            "holdout_seasons": list(self.holdout_seasons),
        }


def split_transaction_records(
    records: list[DecisionReplayRecord],
    holdout_seasons: int = 2,
) -> tuple[list[DecisionReplayRecord], list[DecisionReplayRecord], tuple[int, ...]]:
    """Split replay chronologically so future transaction outcomes stay held out."""

    seasons = sorted({record.season for record in records})
    if holdout_seasons < 1 or len(seasons) <= holdout_seasons:
        return list(records), [], tuple()
    held_out = tuple(seasons[-holdout_seasons:])
    held_out_set = set(held_out)
    train_records = [record for record in records if record.season not in held_out_set]
    validation_records = [record for record in records if record.season in held_out_set]
    return train_records, validation_records, held_out


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


def evaluate_transaction_value_model(
    model: TransactionValueNetwork,
    records: list[DecisionReplayRecord],
) -> dict[str, float | int | None]:
    """Evaluate value predictions against downstream rewards without retraining."""

    predictions: list[float] = []
    targets: list[float] = []
    uncertainties: list[float] = []
    for record in records:
        if record.decision_type not in TRANSACTION_DECISION_TYPES:
            continue
        prediction, uncertainty = model.score(record.features, record.decision_type)
        predictions.append(prediction)
        targets.append(record.reward)
        uncertainties.append(uncertainty)

    if not targets:
        return {
            "records": 0,
            "mae": None,
            "baseline_mae": None,
            "sign_accuracy": None,
            "uncertainty_coverage": None,
        }

    target_mean = mean(targets)
    mae = mean(
        abs(prediction - target)
        for prediction, target in zip(predictions, targets, strict=True)
    )
    baseline_mae = mean(abs(target - target_mean) for target in targets)
    sign_accuracy = mean(
        (prediction - target_mean) * (target - target_mean) >= 0
        for prediction, target in zip(predictions, targets, strict=True)
    )
    uncertainty_coverage = mean(
        abs(prediction - target) <= max(uncertainty, 1.0)
        for prediction, target, uncertainty in zip(
            predictions,
            targets,
            uncertainties,
            strict=True,
        )
    )
    return {
        "records": len(targets),
        "mae": round(mae, 4),
        "baseline_mae": round(baseline_mae, 4),
        "sign_accuracy": round(float(sign_accuracy), 4),
        "uncertainty_coverage": round(float(uncertainty_coverage), 4),
    }


def train_transaction_value_model_with_validation(
    model: TransactionValueNetwork,
    records: list[DecisionReplayRecord],
    epochs: int = 50,
    learning_rate: float = 0.001,
    holdout_seasons: int = 2,
    minimum_validation_records: int = 50,
) -> tuple[float, int, TransactionValueValidation]:
    """Train on earlier seasons and approve only if the holdout beats a baseline."""

    train_records, validation_records, held_out = split_transaction_records(
        records,
        holdout_seasons=holdout_seasons,
    )
    loss, train_count = train_transaction_value_model(
        model=model,
        records=train_records,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    validation = evaluate_transaction_value_model(model, validation_records)
    validation_count = int(validation["records"] or 0)
    mae = validation["mae"]
    baseline_mae = validation["baseline_mae"]
    sign_accuracy = validation["sign_accuracy"]
    uncertainty_coverage = validation["uncertainty_coverage"]
    approved = bool(
        validation_count >= minimum_validation_records
        and mae is not None
        and baseline_mae is not None
        and baseline_mae > 0.0
        and mae < baseline_mae
        and sign_accuracy is not None
        and sign_accuracy >= 0.55
        and uncertainty_coverage is not None
        and uncertainty_coverage >= 0.50
    )
    metrics = TransactionValueValidation(
        train_records=train_count,
        validation_records=validation_count,
        mae=mae,
        baseline_mae=baseline_mae,
        sign_accuracy=sign_accuracy,
        uncertainty_coverage=uncertainty_coverage,
        approved=approved,
        holdout_seasons=held_out,
    )
    return loss, train_count, metrics
