from pathlib import Path

from evolution.training import TrainingResult

DEFAULT_TRAINING_GRAPH_PATH = Path("reports/training_progress.png")


def get_generation_numbers(training_result: TrainingResult) -> list[int]:
    return [
        generation_result.generation_number
        for generation_result in training_result.generation_results
    ]


def get_best_scores(training_result: TrainingResult) -> list[float]:
    return [
        generation_result.best_score for generation_result in training_result.generation_results
    ]


def get_average_scores(training_result: TrainingResult) -> list[float] | None:
    average_scores = []

    for generation_result in training_result.generation_results:
        if generation_result.average_score is None:
            return None

        average_scores.append(generation_result.average_score)

    return average_scores


def save_training_progress_graph(
    training_result: TrainingResult,
    output_path: Path = DEFAULT_TRAINING_GRAPH_PATH,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    generation_numbers = get_generation_numbers(training_result)
    best_scores = get_best_scores(training_result)
    average_scores = get_average_scores(training_result)

    plt.figure()
    plt.plot(
        generation_numbers,
        best_scores,
        marker="o",
        label="Best fitness",
    )

    if average_scores is not None:
        plt.plot(
            generation_numbers,
            average_scores,
            marker="o",
            label="Average fitness",
        )

    plt.title("Training Progress")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path
