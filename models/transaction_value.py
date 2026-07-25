"""Specialized value and uncertainty model for waiver/trade decisions."""

from pathlib import Path

import torch
from torch import nn

from models.league_state_encoder import LEAGUE_STATE_FEATURE_NAMES
from models.manager_policy_nn import MANAGER_FEATURE_COUNT
from models.modular_manager_policy import ModularPolicyFeatures

TRANSACTION_DECISION_TYPES = ("waiver", "trade")
TRANSACTION_VALUE_PATH = Path("data/models/transaction_value_model.pt")


class TransactionValueNetwork(nn.Module):
    """Predict normalized future transaction value and uncertainty.

    The network is deliberately separate from the action policy.  A policy can
    propose an interesting action; this model estimates whether that action is
    likely to create downstream value and how uncertain that estimate is.
    """

    def __init__(
        self,
        player_feature_count: int = MANAGER_FEATURE_COUNT,
        state_feature_count: int = len(LEAGUE_STATE_FEATURE_NAMES),
        hidden_size: int = 64,
    ):
        super().__init__()
        self.player_feature_count = player_feature_count
        self.state_feature_count = state_feature_count
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(player_feature_count + state_feature_count, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict(
            {
                decision_type: nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, 2),
                )
                for decision_type in TRANSACTION_DECISION_TYPES
            }
        )
        self.register_buffer("target_means", torch.zeros(len(TRANSACTION_DECISION_TYPES)))
        self.register_buffer("target_scales", torch.ones(len(TRANSACTION_DECISION_TYPES)))

    def _fit_state_features(self, values: tuple[float, ...]) -> tuple[float, ...]:
        if len(values) < self.state_feature_count:
            return values + (0.0,) * (self.state_feature_count - len(values))
        return values[: self.state_feature_count]

    def _fit_player_features(self, values: tuple[float, ...]) -> tuple[float, ...]:
        """Keep inference compatible with older replay records.

        The modular policy feature schema has grown from 13 to 14 player
        features.  Replay files created before that change should still be
        usable by the value model, so missing values are zero-padded and
        unexpected trailing values are ignored.
        """
        if len(values) < self.player_feature_count:
            return values + (0.0,) * (self.player_feature_count - len(values))
        return values[: self.player_feature_count]

    def forward(
        self,
        player_features: torch.Tensor,
        state_features: torch.Tensor,
        decision_type: str,
    ) -> torch.Tensor:
        if decision_type not in TRANSACTION_DECISION_TYPES:
            raise ValueError(f"Unknown transaction decision type: {decision_type}")
        if player_features.shape[1] < self.player_feature_count:
            padding = torch.zeros(
                (player_features.shape[0], self.player_feature_count - player_features.shape[1]),
                dtype=player_features.dtype,
                device=player_features.device,
            )
            player_features = torch.cat((player_features, padding), dim=1)
        elif player_features.shape[1] > self.player_feature_count:
            player_features = player_features[:, : self.player_feature_count]
        if state_features.shape[1] < self.state_feature_count:
            padding = torch.zeros(
                (state_features.shape[0], self.state_feature_count - state_features.shape[1]),
                dtype=state_features.dtype,
                device=state_features.device,
            )
            state_features = torch.cat((state_features, padding), dim=1)
        elif state_features.shape[1] > self.state_feature_count:
            state_features = state_features[:, : self.state_feature_count]
        encoded = self.encoder(torch.cat((player_features, state_features), dim=1))
        return self.heads[decision_type](encoded)

    def score_normalized(
        self,
        features: ModularPolicyFeatures,
        decision_type: str,
    ) -> tuple[float, float]:
        self.eval()
        players = torch.tensor([features.player_values], dtype=torch.float32)
        states = torch.tensor(
            [self._fit_state_features(features.state_values)],
            dtype=torch.float32,
        )
        with torch.no_grad():
            raw_mean, raw_log_scale = self.forward(players, states, decision_type)[0]
            scale = torch.nn.functional.softplus(raw_log_scale) + 0.05
        return float(raw_mean.item()), float(scale.item())

    def score(
        self,
        features: ModularPolicyFeatures,
        decision_type: str,
    ) -> tuple[float, float]:
        mean, scale = self.score_normalized(features, decision_type)
        index = TRANSACTION_DECISION_TYPES.index(decision_type)
        target_mean = float(self.target_means[index].item())
        target_scale = float(self.target_scales[index].item())
        return (
            target_mean + mean * target_scale,
            scale * target_scale,
        )


def save_transaction_value_model(
    model: TransactionValueNetwork,
    output_path: Path = TRANSACTION_VALUE_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "player_feature_count": model.player_feature_count,
            "state_feature_count": model.state_feature_count,
            "hidden_size": model.hidden_size,
            "state_dict": model.state_dict(),
            "decision_types": list(TRANSACTION_DECISION_TYPES),
        },
        output_path,
    )
    return output_path


def load_transaction_value_model(
    model_path: Path = TRANSACTION_VALUE_PATH,
) -> TransactionValueNetwork:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = TransactionValueNetwork(
        player_feature_count=int(checkpoint["player_feature_count"]),
        state_feature_count=int(checkpoint["state_feature_count"]),
        hidden_size=int(checkpoint["hidden_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
