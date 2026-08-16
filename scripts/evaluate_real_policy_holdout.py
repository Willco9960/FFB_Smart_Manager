from pathlib import Path

from evolution.promotion_gate import evaluate_promotion_gate
from scripts.evaluate_synthetic_policy_real_seasons import (
    ORIGINAL_POLICY_PATH,
    evaluate_policy,
    load_transaction_genome,
)

REAL_POLICY_PATH = Path("data/models/manager_policy_real_seasons.pt")
HOLDOUT_SEASON = 2025
PROMOTION_SEASONS = (2024, 2025)


def main():
    if not REAL_POLICY_PATH.exists():
        raise FileNotFoundError(f"Real-season policy not found: {REAL_POLICY_PATH}")

    transaction_genome = load_transaction_genome()
    original_results = []
    trained_results = []
    for season in PROMOTION_SEASONS:
        original_results.append(evaluate_policy(ORIGINAL_POLICY_PATH, season, transaction_genome))
        trained_results.append(evaluate_policy(REAL_POLICY_PATH, season, transaction_genome))

    decision = evaluate_promotion_gate(
        [result.fitness for result in trained_results],
        [result.fitness for result in original_results],
        [result.wins for result in trained_results],
        [result.wins for result in original_results],
    )
    print("Real-season policy promotion gate")
    for original, trained in zip(original_results, trained_results, strict=True):
        print(
            f"{original.season}: fitness {original.fitness:.2f}->{trained.fitness:.2f}, "
            f"wins {original.wins}->{trained.wins}, "
            f"playoff {original.playoff_seed is not None}->{trained.playoff_seed is not None}"
        )
    print(f"Mean fitness delta: {decision.mean_delta:+.2f}")
    print(f"Bootstrap interval: [{decision.lower_delta:+.2f}, {decision.upper_delta:+.2f}]")
    print(f"Mean wins delta: {decision.wins_delta:+.2f}")
    print(f"PROMOTED: {decision.promoted}")
    if decision.reasons:
        print("Reasons: " + "; ".join(decision.reasons))


if __name__ == "__main__":
    main()
