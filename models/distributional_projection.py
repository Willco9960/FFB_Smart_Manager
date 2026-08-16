"""Quantile and boom-probability projection model.

This is intentionally separate from the legacy point estimator so existing
checkpoints remain loadable while new experiments can use decision-relevant
uncertainty outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from fantasy_engine.projection_dataset import SeasonProjectionExample
from models.draft_projection_nn import FeatureScaler, fit_feature_scaler, transform_features


@dataclass(frozen=True)
class ProjectionDistribution:
    floor: float
    median: float
    ceiling: float
    boom_probability: float


@dataclass(frozen=True)
class DistributionalTrainingResult:
    model: DistributionalProjectionNetwork
    feature_scaler: FeatureScaler
    target_mean: float
    target_standard_deviation: float
    boom_threshold: float
    best_validation_loss: float
    epochs_trained: int


def save_distributional_projection_network(
    result: DistributionalTrainingResult,
    output_path: Path,
    feature_names: tuple[str, ...],
    training_seasons: tuple[int, ...] = (),
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "feature_names": list(feature_names),
        "normalization_means": list(result.feature_scaler.means),
        "normalization_standard_deviations": list(
            result.feature_scaler.standard_deviations
        ),
        "decision_cutoff": f"before_{max(training_seasons)}" if training_seasons else "",
        "model_type": "distributional_quantile_boom_v1",
    }
    import hashlib
    import json

    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    torch.save(
        {
            "input_size": len(result.feature_scaler.means),
            "state_dict": result.model.state_dict(),
            "feature_means": result.feature_scaler.means,
            "feature_standard_deviations": result.feature_scaler.standard_deviations,
            "target_mean": result.target_mean,
            "target_standard_deviation": result.target_standard_deviation,
            "boom_threshold": result.boom_threshold,
            "training_seasons": list(training_seasons),
            "max_training_season": max(training_seasons) if training_seasons else None,
            "feature_manifest": manifest,
            "feature_manifest_digest": digest,
        },
        output_path,
    )
    return output_path


class DistributionalProjectionNetwork(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.quantiles = nn.Linear(32, 3)
        self.boom = nn.Linear(32, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features)
        return self.quantiles(hidden), self.boom(hidden).squeeze(1)


def _pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    error = target - prediction
    return torch.maximum((quantile - 1) * error, quantile * error).mean()


def train_distributional_projection_network(
    training_examples: list[SeasonProjectionExample],
    validation_examples: list[SeasonProjectionExample],
    epochs: int = 300,
    learning_rate: float = 0.005,
    patience: int = 40,
    seed: int = 1,
) -> DistributionalTrainingResult:
    if not training_examples or not validation_examples:
        raise ValueError("Training and validation examples are both required.")
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = fit_feature_scaler(training_examples)
    train_features = transform_features(training_examples, scaler).to(device)
    valid_features = transform_features(validation_examples, scaler).to(device)
    train_targets = torch.tensor(
        [example.target_points for example in training_examples], dtype=torch.float32, device=device
    )
    valid_targets = torch.tensor(
        [example.target_points for example in validation_examples],
        dtype=torch.float32,
        device=device,
    )
    target_mean = float(train_targets.mean().item())
    target_std = max(float(train_targets.std(unbiased=False).item()), 1e-6)
    train_normalized = (train_targets - target_mean) / target_std
    valid_normalized = (valid_targets - target_mean) / target_std
    boom_threshold = float(torch.quantile(train_targets, 0.75).item())
    train_boom = (train_targets >= boom_threshold).float()
    valid_boom = (valid_targets >= boom_threshold).float()
    model = DistributionalProjectionNetwork(len(training_examples[0].features)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best_state = None
    best_loss = float("inf")
    stale = 0
    epochs_trained = 0

    for _epoch in range(epochs):
        epochs_trained += 1
        model.train()
        optimizer.zero_grad()
        quantiles, boom_logits = model(train_features)
        train_loss = sum(
            _pinball_loss(quantiles[:, index], train_normalized, quantile)
            for index, quantile in enumerate((0.1, 0.5, 0.9))
        ) + nn.functional.binary_cross_entropy_with_logits(boom_logits, train_boom)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_quantiles, valid_boom_logits = model(valid_features)
            valid_loss = float(
                (
                    sum(
                        _pinball_loss(valid_quantiles[:, index], valid_normalized, quantile)
                        for index, quantile in enumerate((0.1, 0.5, 0.9))
                    )
                    + nn.functional.binary_cross_entropy_with_logits(valid_boom_logits, valid_boom)
                ).item()
            )
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("Distributional projection network did not produce a state.")
    model.load_state_dict(best_state)
    model.cpu().eval()
    return DistributionalTrainingResult(
        model=model,
        feature_scaler=scaler,
        target_mean=target_mean,
        target_standard_deviation=target_std,
        boom_threshold=boom_threshold,
        best_validation_loss=best_loss,
        epochs_trained=epochs_trained,
    )


def predict_distributions(
    result: DistributionalTrainingResult,
    examples: list[SeasonProjectionExample],
) -> list[ProjectionDistribution]:
    if not examples:
        return []
    features = transform_features(examples, result.feature_scaler)
    with torch.no_grad():
        quantiles, boom_logits = result.model(features)
    distributions = []
    for row, logit in zip(quantiles, boom_logits, strict=True):
        values = [
            float(value * result.target_standard_deviation + result.target_mean) for value in row
        ]
        distributions.append(
            ProjectionDistribution(
                floor=max(0.0, values[0]),
                median=max(0.0, values[1]),
                ceiling=max(0.0, values[2]),
                boom_probability=float(torch.sigmoid(logit).item()),
            )
        )
    return distributions
