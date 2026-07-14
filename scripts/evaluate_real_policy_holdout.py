from pathlib import Path

from scripts.evaluate_synthetic_policy_real_seasons import (
    ORIGINAL_POLICY_PATH,
    evaluate_policy,
    load_transaction_genome,
)

REAL_POLICY_PATH = Path("data/models/manager_policy_real_seasons.pt")
HOLDOUT_SEASON = 2025


def main():
    if not REAL_POLICY_PATH.exists():
        raise FileNotFoundError(f"Real-season policy not found: {REAL_POLICY_PATH}")

    transaction_genome = load_transaction_genome()
    original = evaluate_policy(
        ORIGINAL_POLICY_PATH,
        HOLDOUT_SEASON,
        transaction_genome,
    )
    trained = evaluate_policy(
        REAL_POLICY_PATH,
        HOLDOUT_SEASON,
        transaction_genome,
    )

    print("Real-season policy holdout evaluation")
    print(f"Season: {HOLDOUT_SEASON}")
    print(f"Original fitness: {original.fitness:.2f}")
    print(f"Trained fitness: {trained.fitness:.2f}")
    print(f"Fitness delta: {trained.fitness - original.fitness:+.2f}")
    print(f"Wins: {original.wins}->{trained.wins}")
    print(f"Points for: {original.points_for:.2f}->{trained.points_for:.2f}")
    print(
        "Playoff result: "
        f"{original.playoff_seed is not None}->{trained.playoff_seed is not None}"
    )
    print(f"Champion: {original.champion}->{trained.champion}")


if __name__ == "__main__":
    main()
