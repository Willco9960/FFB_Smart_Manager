from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from fantasy_engine.projection_dataset import SeasonProjectionExample

DEFAULT_MODEL_PATH = Path("data/models/draft_projection_network.pt")


@dataclass(frozen=True)
class FeatureScaler:
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]


@dataclass(frozen=True)
class ProjectionTrainingResult:
    model: "DraftProjectionNetwork"
    feature_scaler: FeatureScaler
    target_mean: float
    target_standard_deviation: float
    best_validation_loss: float
    epochs_trained: int


class DraftProjectionNetwork(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(1)


def select_training_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_feature_scaler(examples: list[SeasonProjectionExample]) -> FeatureScaler:
    feature_count = len(examples[0].features)
    means = []
    standard_deviations = []

    for feature_index in range(feature_count):
        values = [example.features[feature_index] for example in examples]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        standard_deviations.append(max(variance**0.5, 1e-6))

    return FeatureScaler(tuple(means), tuple(standard_deviations))


def transform_features(
    examples: list[SeasonProjectionExample],
    feature_scaler: FeatureScaler,
) -> torch.Tensor:
    rows = []

    for example in examples:
        rows.append(
            [
                (value - mean) / standard_deviation
                for value, mean, standard_deviation in zip(
                    example.features,
                    feature_scaler.means,
                    feature_scaler.standard_deviations,
                    strict=True,
                )
            ]
        )

    return torch.tensor(rows, dtype=torch.float32)


def get_targets(examples: list[SeasonProjectionExample]) -> torch.Tensor:
    return torch.tensor([example.target_points for example in examples], dtype=torch.float32)


def train_projection_network(
    training_examples: list[SeasonProjectionExample],
    validation_examples: list[SeasonProjectionExample],
    epochs: int = 300,
    learning_rate: float = 0.01,
    patience: int = 40,
    seed: int = 1,
) -> ProjectionTrainingResult:
    if not training_examples or not validation_examples:
        raise ValueError("Training and validation examples are both required.")

    torch.manual_seed(seed)
    device = select_training_device()
    feature_scaler = fit_feature_scaler(training_examples)
    training_features = transform_features(training_examples, feature_scaler).to(device)
    validation_features = transform_features(validation_examples, feature_scaler).to(device)
    training_targets = get_targets(training_examples).to(device)
    validation_targets = get_targets(validation_examples).to(device)
    target_mean = float(training_targets.mean().item())
    target_standard_deviation = max(float(training_targets.std(unbiased=False).item()), 1e-6)
    normalized_training_targets = (training_targets - target_mean) / target_standard_deviation
    normalized_validation_targets = (validation_targets - target_mean) / target_standard_deviation
    model = DraftProjectionNetwork(len(training_examples[0].features)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_function = nn.HuberLoss()
    best_state = None
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    epochs_trained = 0

    for _ in range(epochs):
        epochs_trained += 1
        model.train()
        optimizer.zero_grad()
        training_loss = loss_function(model(training_features), normalized_training_targets)
        training_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(validation_features),
                normalized_validation_targets,
            ).item()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("The projection network did not produce a model state.")

    model.load_state_dict(best_state)
    model.to(torch.device("cpu"))

    return ProjectionTrainingResult(
        model=model,
        feature_scaler=feature_scaler,
        target_mean=target_mean,
        target_standard_deviation=target_standard_deviation,
        best_validation_loss=best_validation_loss,
        epochs_trained=epochs_trained,
    )


def predict_points(
    training_result: ProjectionTrainingResult,
    examples: list[SeasonProjectionExample],
) -> list[float]:
    training_result.model.eval()
    features = transform_features(examples, training_result.feature_scaler)

    with torch.no_grad():
        normalized_predictions = training_result.model(features)

    return [
        (prediction.item() * training_result.target_standard_deviation)
        + training_result.target_mean
        for prediction in normalized_predictions
    ]


def calculate_mean_absolute_error(
    predictions: list[float],
    examples: list[SeasonProjectionExample],
) -> float:
    return sum(
        abs(prediction - example.target_points)
        for prediction, example in zip(predictions, examples, strict=True)
    ) / len(examples)


def save_projection_network(
    training_result: ProjectionTrainingResult,
    output_path: Path = DEFAULT_MODEL_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_size": len(training_result.feature_scaler.means),
            "state_dict": training_result.model.state_dict(),
            "feature_means": training_result.feature_scaler.means,
            "feature_standard_deviations": training_result.feature_scaler.standard_deviations,
            "target_mean": training_result.target_mean,
            "target_standard_deviation": training_result.target_standard_deviation,
        },
        output_path,
    )

    return output_path
