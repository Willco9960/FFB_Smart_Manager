from fantasy_engine.weekly_projection_dataset import load_weekly_projection_examples
from models.draft_projection_nn import calculate_mean_absolute_error, predict_points
from models.weekly_projection_nn import (
    convert_weekly_examples,
    save_weekly_projection_network,
    train_weekly_projection_network,
)

TRAINING_SEASONS = [2021, 2022, 2023]
VALIDATION_SEASON = 2024
TEST_SEASON = 2025


def main():
    examples = load_weekly_projection_examples(
        TRAINING_SEASONS + [VALIDATION_SEASON, TEST_SEASON]
    )
    training_examples = [example for example in examples if example.season in TRAINING_SEASONS]
    validation_examples = [example for example in examples if example.season == VALIDATION_SEASON]
    test_examples = [example for example in examples if example.season == TEST_SEASON]
    training_result = train_weekly_projection_network(
        training_examples,
        validation_examples,
        epochs=300,
        learning_rate=0.005,
        patience=40,
    )
    test_projection_examples = convert_weekly_examples(test_examples)
    test_predictions = predict_points(training_result, test_projection_examples)
    test_mae = calculate_mean_absolute_error(
        test_predictions,
        test_projection_examples,
    )
    model_path = save_weekly_projection_network(training_result)

    print("Weekly projection network training complete")
    print(f"Training examples: {len(training_examples)}")
    print(f"Validation examples: {len(validation_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print(f"Test MAE: {test_mae:.2f} fantasy points")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
