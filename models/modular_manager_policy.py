"""Modular manager policy with shared state and specialized decision heads."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from models.feature_manifest import create_feature_manifest, validate_checkpoint_manifest
from models.league_state_encoder import LEAGUE_STATE_FEATURE_NAMES, create_league_state_features
from models.manager_policy_nn import MANAGER_FEATURE_COUNT, create_draft_action_features

DECISION_TYPES = ("draft", "lineup", "waiver", "trade")
MODULAR_POLICY_PATH = Path("data/models/modular_manager_policy.pt")


@dataclass(frozen=True)
class ModularPolicyFeatures:
    player_values: tuple[float, ...]
    state_values: tuple[float, ...]


def create_modular_policy_features(
    player,
    team,
    available_players,
    current_week: int = 0,
    projection_uncertainty: float = 0.0,
    opponent_strength: float = 0.0,
    standing_win_rate: float = 0.0,
    playoff_probability: float = 0.0,
    projection_floor: float = 0.0,
    projection_median: float = 0.0,
    projection_ceiling: float = 0.0,
    boom_probability: float = 0.0,
) -> ModularPolicyFeatures:
    """Adapt existing action features into the shared modular representation."""

    return ModularPolicyFeatures(
        player_values=create_draft_action_features(
            player,
            team,
            available_players,
        ).values,
        state_values=create_league_state_features(
            team,
            available_players,
            current_week=current_week,
            projection_uncertainty=projection_uncertainty,
            opponent_strength=opponent_strength,
            standing_win_rate=standing_win_rate,
            playoff_probability=playoff_probability,
            projection_floor=projection_floor,
            projection_median=projection_median,
            projection_ceiling=projection_ceiling,
            boom_probability=boom_probability,
        ),
    )


class ModularManagerPolicyNetwork(nn.Module):
    """A small shared encoder with one head per fantasy decision type.

    This intentionally remains compact.  The simulator has far fewer examples
    than a language model, so a large Transformer would overfit rather than
    improve the manager.
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
        self.player_encoder = nn.Sequential(
            nn.Linear(player_feature_count, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_feature_count, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
        )
        combined_size = (hidden_size // 2) + (hidden_size // 4)
        self.decision_heads = nn.ModuleDict(
            {
                decision_type: nn.Sequential(
                    nn.Linear(combined_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, 1),
                )
                for decision_type in DECISION_TYPES
            }
        )
        self.value_head = nn.Sequential(
            nn.Linear(combined_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def encode(self, features: ModularPolicyFeatures) -> torch.Tensor:
        device = next(self.parameters()).device
        player = torch.tensor([features.player_values], dtype=torch.float32, device=device)
        state = torch.tensor(
            [self._fit_state_features(features.state_values)],
            dtype=torch.float32,
            device=device,
        )
        return torch.cat((self.player_encoder(player), self.state_encoder(state)), dim=1)

    def _fit_state_features(self, values: tuple[float, ...]) -> tuple[float, ...]:
        """Keep checkpoints trained before context features backward compatible."""

        if len(values) < self.state_feature_count:
            return values + (0.0,) * (self.state_feature_count - len(values))
        return values[: self.state_feature_count]

    def forward(
        self,
        player_features: torch.Tensor,
        state_features: torch.Tensor,
        decision_type: str = "draft",
    ) -> torch.Tensor:
        if decision_type not in DECISION_TYPES:
            raise ValueError(f"Unknown decision type: {decision_type}")
        if state_features.shape[1] < self.state_feature_count:
            padding = torch.zeros(
                (state_features.shape[0], self.state_feature_count - state_features.shape[1]),
                dtype=state_features.dtype,
                device=state_features.device,
            )
            state_features = torch.cat((state_features, padding), dim=1)
        elif state_features.shape[1] > self.state_feature_count:
            state_features = state_features[:, : self.state_feature_count]
        encoded = torch.cat(
            (self.player_encoder(player_features), self.state_encoder(state_features)),
            dim=1,
        )
        return self.decision_heads[decision_type](encoded).squeeze(1)

    def score_decision(
        self,
        features: ModularPolicyFeatures,
        decision_type: str,
    ) -> float:
        return self.score_decisions([features], decision_type)[0]

    def score_decisions(
        self,
        features: list[ModularPolicyFeatures],
        decision_type: str,
    ) -> list[float]:
        """Score many candidate actions in one forward pass.

        Drafting evaluates the entire available player pool for every pick.
        Keeping that work batched avoids creating and executing one tiny
        tensor operation per player while preserving the existing scalar API.
        """

        if decision_type not in DECISION_TYPES:
            raise ValueError(f"Unknown decision type: {decision_type}")
        if not features:
            return []

        self.eval()
        player_values = torch.tensor(
            [item.player_values for item in features],
            dtype=torch.float32,
            device=next(self.parameters()).device,
        )
        state_values = torch.tensor(
            [self._fit_state_features(item.state_values) for item in features],
            dtype=torch.float32,
            device=next(self.parameters()).device,
        )
        with torch.no_grad():
            scores = self.forward(
                player_features=player_values,
                state_features=state_values,
                decision_type=decision_type,
            )

        return [float(score) for score in scores.tolist()]

    def score_draft_action(self, features: ModularPolicyFeatures) -> float:
        return self.score_decision(features, "draft")

    def score_lineup_action(self, features: ModularPolicyFeatures) -> float:
        return self.score_decision(features, "lineup")

    def score_waiver_action(self, features: ModularPolicyFeatures) -> float:
        return self.score_decision(features, "waiver")

    def score_trade_action(self, features: ModularPolicyFeatures) -> float:
        return self.score_decision(features, "trade")

    def score_action(self, features: ModularPolicyFeatures) -> float:
        """Compatibility alias used by the existing draft agent interface."""

        return self.score_draft_action(features)

    def estimate_value(self, features: ModularPolicyFeatures) -> float:
        self.eval()
        with torch.no_grad():
            return float(self.value_head(self.encode(features)).item())


def save_modular_policy_network(
    model: ModularManagerPolicyNetwork,
    output_path: Path = MODULAR_POLICY_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = create_feature_manifest(
        feature_names=tuple(LEAGUE_STATE_FEATURE_NAMES),
        decision_cutoff="pre-decision",
    )
    torch.save(
        {
            "player_feature_count": model.player_feature_count,
            "state_feature_count": model.state_feature_count,
            "hidden_size": model.hidden_size,
            "state_dict": model.state_dict(),
            "decision_types": list(DECISION_TYPES),
            "feature_manifest": manifest.to_dict(),
            "feature_manifest_digest": manifest.digest(),
        },
        output_path,
    )
    return output_path


def load_modular_policy_network(
    model_path: Path = MODULAR_POLICY_PATH,
) -> ModularManagerPolicyNetwork:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if "feature_manifest" in checkpoint:
        validate_checkpoint_manifest(checkpoint)
    model = ModularManagerPolicyNetwork(
        player_feature_count=int(checkpoint["player_feature_count"]),
        state_feature_count=int(checkpoint["state_feature_count"]),
        hidden_size=int(checkpoint["hidden_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
