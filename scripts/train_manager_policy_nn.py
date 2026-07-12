from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from agents.genome_draft_agent import GenomeDraftAgent
from evolution.genome import create_random_genome
from fantasy_engine.draft import get_snake_draft_order
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.team import Team
from models.draft_projection_nn import DEFAULT_MODEL_PATH
from models.manager_policy_nn import (
    MANAGER_FEATURE_COUNT,
    ManagerPolicyNetwork,
    create_draft_action_features,
    save_manager_policy_network,
)
from models.projection_service import load_neural_projection_service

OUTPUT_PATH = Path("data/models/manager_policy_network.pt")
TRAINING_EPISODES = 20
TRAINING_EPOCHS = 200


@dataclass(frozen=True)
class ImitationExample:
    features: tuple[float, ...]
    target_score: float


def create_training_league() -> League:
    teams = [Team(name=f"Policy Team {number}") for number in range(1, 11)]
    league = League(
        name="Manager Policy Training League",
        teams=teams,
        available_players=load_leakage_safe_player_pool(2020, 2021)[:250],
    )
    projection_service = load_neural_projection_service(DEFAULT_MODEL_PATH)

    if projection_service is not None:
        league = projection_service.project_league(league)

    return league


def collect_imitation_examples(
    league: League,
    teacher: GenomeDraftAgent,
    episodes: int = TRAINING_EPISODES,
    rounds: int = 16,
) -> list[ImitationExample]:
    examples = []

    for _episode in range(episodes):
        teams = [Team(name=team.name) for team in league.teams]
        episode_league = League(
            name=league.name,
            teams=teams,
            available_players=list(league.available_players),
        )

        for round_number in range(1, rounds + 1):
            for team in get_snake_draft_order(episode_league.teams, round_number):
                available_players = episode_league.available_players
                teacher_scores = [
                    teacher.score_player(player, available_players)
                    for player in available_players
                ]
                maximum_score = max(max(teacher_scores), 1.0)
                examples.extend(
                    ImitationExample(
                        features=create_draft_action_features(
                            player,
                            team,
                            available_players,
                        ).values,
                        target_score=score / maximum_score,
                    )
                    for player, score in zip(available_players, teacher_scores, strict=True)
                )
                selected_player = teacher.choose_player(
                    available_players=available_players,
                    team=team,
                    league=episode_league,
                )
                team.add_player(selected_player)
                available_players.remove(selected_player)

    return examples


def train_manager_policy(
    examples: list[ImitationExample],
    epochs: int = TRAINING_EPOCHS,
) -> tuple[ManagerPolicyNetwork, float]:
    features = torch.tensor([example.features for example in examples], dtype=torch.float32)
    targets = torch.tensor([example.target_score for example in examples], dtype=torch.float32)
    model = ManagerPolicyNetwork(input_size=MANAGER_FEATURE_COUNT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    loss_function = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

    return model, float(loss.item())


def main():
    genome = create_random_genome(seed=2021)
    teacher = GenomeDraftAgent(genome=genome)
    examples = collect_imitation_examples(create_training_league(), teacher)
    model, final_loss = train_manager_policy(examples)
    save_manager_policy_network(model, OUTPUT_PATH)

    print("Manager policy imitation training complete")
    print(f"Training examples: {len(examples)}")
    print(f"Final imitation loss: {final_loss:.6f}")
    print(f"Model saved to: {OUTPUT_PATH}")
    print("The next training stage will use complete-season rewards instead of teacher labels.")


if __name__ == "__main__":
    main()
