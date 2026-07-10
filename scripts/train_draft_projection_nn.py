from fantasy_engine.projection_dataset import get_feature_names, load_projection_examples
from models.draft_projection_nn import (
    calculate_mean_absolute_error,
    predict_points,
    save_projection_network,
    select_training_device,
    train_projection_network,
)

TRAINING_SEASONS = [2020, 2021, 2022, 2023]
VALIDATION_SEASON = 2024
TEST_SEASON = 2025


def main():
    examples = load_projection_examples(TRAINING_SEASONS + [VALIDATION_SEASON, TEST_SEASON])
    training_examples = [example for example in examples if example.season in TRAINING_SEASONS]
    validation_examples = [example for example in examples if example.season == VALIDATION_SEASON]
    test_examples = [example for example in examples if example.season == TEST_SEASON]
    training_result = train_projection_network(training_examples, validation_examples)
    test_predictions = predict_points(training_result, test_examples)
    test_mae = calculate_mean_absolute_error(test_predictions, test_examples)
    model_path = save_projection_network(training_result)

    print(f"Training device: {select_training_device()}")
    print(f"Features: {', '.join(get_feature_names())}")
    print(f"Training examples: {len(training_examples)}")
    print(f"Validation examples: {len(validation_examples)}")
    print(f"Test examples: {len(test_examples)}")
    print(f"Best validation loss: {training_result.best_validation_loss:.4f}")
    print(f"Test MAE: {test_mae:.2f} fantasy points")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
