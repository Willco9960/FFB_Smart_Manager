"""Run the policy warm-start gate before an expensive self-play run."""

from __future__ import annotations

import argparse
from pathlib import Path

from evolution.pretraining import build_manager_teacher_examples, run_manager_pretraining
from fantasy_engine.data_availability import validate_training_seasons
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.team import Team
from models.modular_manager_policy import ModularManagerPolicyNetwork, save_modular_policy_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/pretrained_manager_policy.pt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is unavailable.")
    manifests = validate_training_seasons((args.season - 1, args.season))
    players = load_leakage_safe_player_pool(
        projection_season=args.season - 1,
        actual_season=args.season,
        include_special_teams=True,
    )
    league = League(
        name=f"Pretraining gate {args.season}",
        teams=[Team(name=f"Team {index}") for index in range(1, 11)],
        available_players=players,
    )
    examples = build_manager_teacher_examples(
        league,
        episodes=args.episodes,
        rounds=args.rounds,
    )
    model = ModularManagerPolicyNetwork()
    result = run_manager_pretraining(
        model,
        examples,
        behavior_epochs=args.epochs,
        device=args.device,
    )
    if not result.approved:
        raise SystemExit("Pretraining gate failed: non-finite loss.")
    save_modular_policy_network(model, args.output)
    print("Manager pretraining gate passed")
    print(f"Season: {args.season}")
    print(f"Data rows: {manifests[0].row_count}")
    print(f"Examples: {result.example_count}")
    print(f"Decision heads: {dict(result.decision_type_counts)}")
    print(f"Behavior loss: {result.behavior_loss:.6f}")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
