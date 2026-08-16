import argparse

from fantasy_engine.historical_seasons import EARLIEST_RELIABLE_SEASON
from fantasy_engine.projection_dataset import get_feature_names, load_projection_examples
from models.distributional_projection import (
    predict_distributions,
    save_distributional_projection_network,
    train_distributional_projection_network,
)
from models.draft_projection_nn import (
    calculate_mean_absolute_error,
    predict_points,
    save_projection_network,
    select_training_device,
    train_projection_network,
)
from models.projection_metrics import (
    quantile_coverage,
    spearman_rank_correlation,
    top_k_hit_rate,
)

TRAINING_START_SEASON = EARLIEST_RELIABLE_SEASON
TRAINING_END_SEASON = 2019
VALIDATION_SEASON = 2020
TEST_SEASON = 2021


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a leakage-safe draft projection network.")
    parser.add_argument("--train-start", type=int, default=TRAINING_START_SEASON)
    parser.add_argument("--train-end", type=int, default=TRAINING_END_SEASON)
    parser.add_argument("--validation-season", type=int, default=VALIDATION_SEASON)
    parser.add_argument("--test-season", type=int, default=TEST_SEASON)
    parser.add_argument(
        "--distributional",
        action="store_true",
        help="Also train and save floor/median/ceiling/boom outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    training_seasons = list(range(args.train_start, args.train_end + 1))

    if not training_seasons:
        raise ValueError("At least one draft projection training season is required.")

    if args.train_end >= args.validation_season:
        raise ValueError("Training seasons must end before the validation season.")

    if args.validation_season >= args.test_season:
        raise ValueError("Validation season must be before the test season.")

    target_seasons = training_seasons + [args.validation_season, args.test_season]
    examples = load_projection_examples(target_seasons)
    training_examples = [example for example in examples if example.season in training_seasons]
    validation_examples = [
        example for example in examples if example.season == args.validation_season
    ]
    test_examples = [example for example in examples if example.season == args.test_season]
    training_result = train_projection_network(training_examples, validation_examples)
    test_predictions = predict_points(training_result, test_examples)
    test_mae = calculate_mean_absolute_error(test_predictions, test_examples)
    model_path = save_projection_network(
        training_result,
        training_seasons=tuple(training_seasons),
    )

    print(f"Training device: {select_training_device()}")
    print(f"Training seasons: {tuple(training_seasons)}")
    print(f"Validation season: {args.validation_season}")
    print(f"Test season: {args.test_season}")
    print(f"Features: {', '.join(get_feature_names())}")
    print(f"Training examples: {len(training_examples)}")
    print(f"Validation examples: {len(validation_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print(f"Best validation loss: {training_result.best_validation_loss:.4f}")
    print(f"Test MAE: {test_mae:.2f} fantasy points")
    print(f"Model saved to: {model_path}")

    if args.distributional:
        distributional_result = train_distributional_projection_network(
            training_examples,
            validation_examples,
        )
        distributions = predict_distributions(distributional_result, test_examples)
        targets = [example.target_points for example in test_examples]
        medians = [prediction.median for prediction in distributions]
        floor = [prediction.floor for prediction in distributions]
        ceiling = [prediction.ceiling for prediction in distributions]
        output_path = save_distributional_projection_network(
            distributional_result,
            model_path.with_name("draft_projection_distributional.pt"),
            tuple(get_feature_names()),
            tuple(training_seasons),
        )
        print(
            "Distributional median rank correlation: "
            f"{spearman_rank_correlation(medians, targets):.3f}"
        )
        print(f"Distributional top-24 hit rate: {top_k_hit_rate(medians, targets, 24):.3f}")
        print(f"Distributional 10-90 coverage: {quantile_coverage(floor, ceiling, targets):.3f}")
        print(f"Distributional model saved to: {output_path}")


if __name__ == "__main__":
    main()
