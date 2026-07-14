import argparse

from fantasy_engine.historical_seasons import EARLIEST_RELIABLE_SEASON
from fantasy_engine.weekly_projection_dataset import load_weekly_projection_examples
from models.draft_projection_nn import calculate_mean_absolute_error, predict_points
from models.weekly_projection_nn import (
    calibrate_neural_weights_by_position,
    convert_weekly_examples,
    predict_calibrated_weekly_points,
    save_weekly_projection_network,
    train_weekly_projection_network,
)

TRAINING_START_SEASON = EARLIEST_RELIABLE_SEASON
TRAINING_END_SEASON = 2019
VALIDATION_SEASON = 2020
TEST_SEASON = 2021


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a leakage-safe weekly projection network.")
    parser.add_argument("--train-start", type=int, default=TRAINING_START_SEASON)
    parser.add_argument("--train-end", type=int, default=TRAINING_END_SEASON)
    parser.add_argument("--validation-season", type=int, default=VALIDATION_SEASON)
    parser.add_argument("--test-season", type=int, default=TEST_SEASON)
    return parser.parse_args()


def main():
    args = parse_args()
    training_seasons = list(range(args.train_start, args.train_end + 1))

    if not training_seasons:
        raise ValueError("At least one weekly projection training season is required.")

    if args.train_end >= args.validation_season:
        raise ValueError("Training seasons must end before the validation season.")

    if args.validation_season >= args.test_season:
        raise ValueError("Validation season must be before the test season.")

    target_seasons = training_seasons + [args.validation_season, args.test_season]
    examples = load_weekly_projection_examples(target_seasons)
    training_examples = [example for example in examples if example.season in training_seasons]
    validation_examples = [
        example for example in examples if example.season == args.validation_season
    ]
    test_examples = [example for example in examples if example.season == args.test_season]
    training_result = train_weekly_projection_network(
        training_examples,
        validation_examples,
        epochs=300,
        learning_rate=0.005,
        patience=40,
    )
    neural_weights_by_position = calibrate_neural_weights_by_position(
        training_result,
        validation_examples,
    )
    validation_projection_examples = convert_weekly_examples(validation_examples)
    validation_neural_predictions = predict_points(
        training_result,
        validation_projection_examples,
    )
    validation_blended_predictions = predict_calibrated_weekly_points(
        training_result,
        validation_examples,
        neural_weights_by_position,
    )
    validation_neural_mae = calculate_mean_absolute_error(
        validation_neural_predictions,
        validation_projection_examples,
    )
    validation_blended_mae = calculate_mean_absolute_error(
        validation_blended_predictions,
        validation_projection_examples,
    )
    test_projection_examples = convert_weekly_examples(test_examples)
    test_predictions = predict_calibrated_weekly_points(
        training_result,
        test_examples,
        neural_weights_by_position,
    )
    test_mae = calculate_mean_absolute_error(
        test_predictions,
        test_projection_examples,
    )
    model_path = save_weekly_projection_network(
        training_result,
        neural_weights_by_position=neural_weights_by_position,
        training_seasons=tuple(training_seasons),
    )

    print("Weekly projection network training complete")
    print(f"Training seasons: {tuple(training_seasons)}")
    print(f"Validation season: {args.validation_season}")
    print(f"Test season: {args.test_season}")
    print(f"Training examples: {len(training_examples)}")
    print(f"Validation examples: {len(validation_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print(f"Test MAE: {test_mae:.2f} fantasy points")
    print(f"Validation neural MAE: {validation_neural_mae:.2f} fantasy points")
    print(f"Validation blended MAE: {validation_blended_mae:.2f} fantasy points")
    print(f"Neural weights by position: {neural_weights_by_position}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
