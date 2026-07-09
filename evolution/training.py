from dataclasses import dataclass
from pathlib import Path

from evolution.genome import DraftStrategyGenome
from evolution.population import (
    create_agent_population,
    create_next_generation,
    evaluate_population,
    rank_evaluated_agents,
    select_top_genomes,
)
from fantasy_engine.league import League

DEFAULT_POPULATION_SIZE = 100
DEFAULT_GENERATION_COUNT = 20
DEFAULT_SELECTION_COUNT = 10
DEFAULT_MUTATION_STRENGTH = 0.1
DEFAULT_OUTPUT_PATH = Path("data/evolution/best_genome.json")


@dataclass
class GenerationResult:
    generation_number: int
    best_score: float
    best_genome: DraftStrategyGenome


@dataclass
class TrainingResult:
    generation_results: list[GenerationResult]
    best_score: float
    best_genome: DraftStrategyGenome


def run_training_experiment(
    league: League,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generation_count: int = DEFAULT_GENERATION_COUNT,
    selection_count: int = DEFAULT_SELECTION_COUNT,
    mutation_strength: float = DEFAULT_MUTATION_STRENGTH,
    seed: int = 1,
    rounds: int = 16,
) -> TrainingResult:
    agents = create_agent_population(
        population_size=population_size,
        seed=seed,
    )
    generation_results = []
    best_score = 0.0
    best_genome = agents[0].genome

    for generation_number in range(1, generation_count + 1):
        evaluated_agents = evaluate_population(
            agents=agents,
            league=league,
            rounds=rounds,
        )
        ranked_agents = rank_evaluated_agents(evaluated_agents)
        generation_best_agent = ranked_agents[0]

        generation_result = GenerationResult(
            generation_number=generation_number,
            best_score=generation_best_agent.fitness_score,
            best_genome=generation_best_agent.genome,
        )
        generation_results.append(generation_result)

        if generation_best_agent.fitness_score > best_score:
            best_score = generation_best_agent.fitness_score
            best_genome = generation_best_agent.genome

        selected_genomes = select_top_genomes(
            evaluated_agents=evaluated_agents,
            selection_count=selection_count,
        )
        agents = create_next_generation(
            selected_genomes=selected_genomes,
            population_size=population_size,
            seed=seed + generation_number,
            mutation_strength=mutation_strength,
        )

    return TrainingResult(
        generation_results=generation_results,
        best_score=best_score,
        best_genome=best_genome,
    )


def format_training_log(training_result: TrainingResult) -> str:
    lines = ["Evolution training complete"]

    for generation_result in training_result.generation_results:
        lines.append(
            f"Generation {generation_result.generation_number}: "
            f"best score = {generation_result.best_score}"
        )

    lines.append(f"Overall best score: {training_result.best_score}")

    return "\n".join(lines)


def save_best_genome(
    training_result: TrainingResult,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(training_result.best_genome.to_json(), encoding="utf-8")

    return output_path


def run_and_save_training_experiment(
    league: League,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generation_count: int = DEFAULT_GENERATION_COUNT,
    selection_count: int = DEFAULT_SELECTION_COUNT,
    mutation_strength: float = DEFAULT_MUTATION_STRENGTH,
    seed: int = 1,
    rounds: int = 16,
) -> TrainingResult:
    training_result = run_training_experiment(
        league=league,
        population_size=population_size,
        generation_count=generation_count,
        selection_count=selection_count,
        mutation_strength=mutation_strength,
        seed=seed,
        rounds=rounds,
    )
    save_best_genome(training_result, output_path)

    return training_result
