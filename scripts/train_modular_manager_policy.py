"""Warm-start and self-play train the modular manager policy."""

import argparse
from pathlib import Path

from agents.genome_draft_agent import GenomeDraftAgent
from evolution.genome import create_random_genome
from evolution.modular_behavior_cloning import (
    ModularImitationExample,
    train_modular_behavior_policy,
)
from evolution.modular_policy_training import train_modular_policy_self_play
from fantasy_engine.draft import get_snake_draft_order
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import load_weekly_performances
from models.modular_manager_policy import (
    ModularManagerPolicyNetwork,
    create_modular_policy_features,
    save_modular_policy_network,
)

OUTPUT_PATH = Path("data/models/modular_manager_policy.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2021)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--selection", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def create_training_league(season: int) -> League:
    return League(
        name=f"Modular Training League {season}",
        teams=[Team(name=f"Modular Team {number}") for number in range(1, 11)],
        available_players=load_leakage_safe_player_pool(
            projection_season=season - 1,
            actual_season=season,
            include_special_teams=True,
        )[:250],
    )


def collect_draft_examples(
    league: League,
    teacher: GenomeDraftAgent,
    episodes: int = 2,
    rounds: int = 16,
) -> list[ModularImitationExample]:
    examples = []
    for _ in range(episodes):
        episode_league = League(
            name=league.name,
            teams=[Team(name=team.name) for team in league.teams],
            available_players=list(league.available_players),
        )
        for round_number in range(1, rounds + 1):
            for team in get_snake_draft_order(episode_league.teams, round_number):
                available = episode_league.available_players
                scores = [teacher.score_player(player, available) for player in available]
                maximum = max(max(scores), 1.0)
                examples.extend(
                    ModularImitationExample(
                        features=create_modular_policy_features(player, team, available),
                        target_score=score / maximum,
                    )
                    for player, score in zip(available, scores, strict=True)
                )
                selected = teacher.choose_player(available, team, episode_league)
                team.add_player(selected)
                available.remove(selected)
    return examples


def main() -> None:
    args = parse_args()
    model = ModularManagerPolicyNetwork()
    teacher = GenomeDraftAgent(create_random_genome(seed=2021))
    first_league = create_training_league(args.start_season)
    examples = collect_draft_examples(first_league, teacher)
    imitation_loss = train_modular_behavior_policy(model, examples, epochs=args.epochs)

    scenarios = [
        (
            create_training_league(season),
            load_weekly_performances(season, include_special_teams=True),
        )
        for season in range(args.start_season, args.end_season + 1)
    ]
    trained_model, history = train_modular_policy_self_play(
        initial_policy=model,
        scenarios=scenarios,
        transaction_genome=teacher.genome,
        population_size=args.population,
        generations=args.generations,
        selection_count=args.selection,
        lineup_rules=ESPN_OFFENSIVE_LINEUP_RULES,
    )
    model_path = save_modular_policy_network(trained_model, args.output)
    print("Modular manager policy training complete")
    print(f"Behavioral examples: {len(examples)}")
    print(f"Behavioral cloning loss: {imitation_loss:.6f}")
    print(f"Self-play best fitness by generation: {history}")
    print(f"Policy saved to: {model_path}")


if __name__ == "__main__":
    main()
