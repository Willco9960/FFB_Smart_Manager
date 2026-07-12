from dataclasses import dataclass

import torch
from torch import nn

from fantasy_engine.player import Player
from fantasy_engine.team import Team

POSITION_ORDER = ("QB", "RB", "WR", "TE")
MANAGER_FEATURE_COUNT = 14


@dataclass(frozen=True)
class ManagerActionFeatures:
    values: tuple[float, ...]


def create_draft_action_features(
    player: Player,
    team: Team,
    available_players: list[Player],
) -> ManagerActionFeatures:
    position_counts = {
        position: sum(roster_player.position == position for roster_player in team.roster)
        for position in POSITION_ORDER
    }
    same_position_players = [
        available_player
        for available_player in available_players
        if available_player.position == player.position
    ]
    highest_available_projection = max(
        (available_player.projected_score for available_player in available_players),
        default=1.0,
    )
    position_rank = 1 + sum(
        available_player.projected_score > player.projected_score
        for available_player in same_position_players
    )

    values = (
        player.projected_score / 500.0,
        player.projected_score / max(highest_available_projection, 1.0),
        float(position_rank) / max(float(len(same_position_players)), 1.0),
        float(len(same_position_players)) / 100.0,
        float(len(team.roster)) / 16.0,
        team.projected_score() / 8000.0,
        *tuple(float(position_counts[position]) / 16.0 for position in POSITION_ORDER),
        *tuple(float(player.position == position) for position in POSITION_ORDER),
    )

    return ManagerActionFeatures(values=values)


class ManagerPolicyNetwork(nn.Module):
    def __init__(self, input_size: int = MANAGER_FEATURE_COUNT):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(1)

    def score_action(self, features: ManagerActionFeatures) -> float:
        self.eval()
        with torch.no_grad():
            tensor = torch.tensor([features.values], dtype=torch.float32)
            return float(self(tensor).item())


def save_manager_policy_network(
    model: ManagerPolicyNetwork,
    output_path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_size": MANAGER_FEATURE_COUNT,
            "state_dict": model.state_dict(),
        },
        output_path,
    )


def load_manager_policy_network(model_path) -> ManagerPolicyNetwork:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = ManagerPolicyNetwork(input_size=checkpoint["input_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
