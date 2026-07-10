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
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot
from fantasy_engine.player import Player

DEFAULT_POPULATION_SIZE = 100
DEFAULT_GENERATION_COUNT = 20
DEFAULT_SELECTION_COUNT = 10
DEFAULT_MUTATION_STRENGTH = 0.1
DEFAULT_FINAL_MUTATION_STRENGTH = 0.03
DEFAULT_OUTPUT_PATH = Path("data/evolution/best_genome.json")


@dataclass
class GenerationResult:
    generation_number: int
    best_score: float
    best_genome: DraftStrategyGenome
    winning_team_name: str
    winning_roster: list[Player]
    average_score: float | None = None


@dataclass
class TrainingResult:
    generation_results: list[GenerationResult]
    best_score: float
    best_genome: DraftStrategyGenome


def calculate_generation_mutation_strength(
    generation_number: int,
    generation_count: int,
    initial_mutation_strength: float = DEFAULT_MUTATION_STRENGTH,
    final_mutation_strength: float = DEFAULT_FINAL_MUTATION_STRENGTH,
) -> float:
    if generation_count <= 1:
        return round(final_mutation_strength, 4)

    progress = (generation_number - 1) / (generation_count - 1)
    mutation_strength = initial_mutation_strength - (
        (initial_mutation_strength - final_mutation_strength) * progress
    )

    return round(mutation_strength, 4)


def run_training_experiment(
    league: League,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generation_count: int = DEFAULT_GENERATION_COUNT,
    selection_count: int = DEFAULT_SELECTION_COUNT,
    mutation_strength: float = DEFAULT_MUTATION_STRENGTH,
    final_mutation_strength: float = DEFAULT_FINAL_MUTATION_STRENGTH,
    seed: int = 1,
    rounds: int = 16,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
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
            lineup_rules=lineup_rules,
            seed=seed + (generation_number * population_size),
        )
        ranked_agents = rank_evaluated_agents(evaluated_agents)
        generation_best_agent = ranked_agents[0]
        average_score = round(
            sum(evaluated_agent.fitness_score for evaluated_agent in evaluated_agents)
            / len(evaluated_agents),
            2,
        )

        generation_result = GenerationResult(
            generation_number=generation_number,
            best_score=generation_best_agent.fitness_score,
            best_genome=generation_best_agent.genome,
            winning_team_name=generation_best_agent.winning_team_name,
            winning_roster=generation_best_agent.winning_roster,
            average_score=average_score,
        )
        generation_results.append(generation_result)

        if generation_best_agent.fitness_score > best_score:
            best_score = generation_best_agent.fitness_score
            best_genome = generation_best_agent.genome

        selected_genomes = select_top_genomes(
            evaluated_agents=evaluated_agents,
            selection_count=selection_count,
        )

        generation_mutation_strength = calculate_generation_mutation_strength(
            generation_number=generation_number,
            generation_count=generation_count,
            initial_mutation_strength=mutation_strength,
            final_mutation_strength=final_mutation_strength,
        )

        agents = create_next_generation(
            selected_genomes=selected_genomes,
            population_size=population_size,
            seed=seed + generation_number,
            mutation_strength=generation_mutation_strength,
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
    final_mutation_strength: float = DEFAULT_FINAL_MUTATION_STRENGTH,
    seed: int = 1,
    rounds: int = 16,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
) -> TrainingResult:
    training_result = run_training_experiment(
        league=league,
        population_size=population_size,
        generation_count=generation_count,
        selection_count=selection_count,
        mutation_strength=mutation_strength,
        final_mutation_strength=final_mutation_strength,
        seed=seed,
        rounds=rounds,
        lineup_rules=lineup_rules,
    )
    save_best_genome(training_result, output_path)

    return training_result


def get_roster_signature(roster: list[Player]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((player.name, player.position, player.team) for player in roster))


def get_generation_result(
    training_result: TrainingResult,
    generation_number: int,
) -> GenerationResult:
    for generation_result in training_result.generation_results:
        if generation_result.generation_number == generation_number:
            return generation_result

    raise ValueError(f"Generation {generation_number} was not found.")


def have_same_winning_roster(
    training_result: TrainingResult,
    generation_numbers: list[int],
) -> bool:
    if not generation_numbers:
        raise ValueError("At least one generation number is required.")

    first_generation_result = get_generation_result(
        training_result,
        generation_numbers[0],
    )
    first_signature = get_roster_signature(first_generation_result.winning_roster)

    for generation_number in generation_numbers[1:]:
        generation_result = get_generation_result(
            training_result,
            generation_number,
        )
        signature = get_roster_signature(generation_result.winning_roster)

        if signature != first_signature:
            return False

    return True


def format_roster(player_roster: list[Player]) -> str:
    lines = []

    for player in player_roster:
        lines.append(
            f"{player.name} "
            f"({player.position}, {player.team}) - "
            f"projection {player.projected_score}, "
            f"actual {player.actual_score}"
        )

    return "\n".join(lines)


def format_generation_roster_report(
    training_result: TrainingResult,
    generation_numbers: list[int],
) -> str:
    lines = []

    for generation_number in generation_numbers:
        generation_result = get_generation_result(
            training_result,
            generation_number,
        )
        lines.append(
            f"Generation {generation_number}: "
            f"{generation_result.winning_team_name}, "
            f"score {generation_result.best_score}"
        )
        lines.append(format_roster(generation_result.winning_roster))
        lines.append("")

    same_roster = have_same_winning_roster(
        training_result=training_result,
        generation_numbers=generation_numbers,
    )
    lines.append(f"Same winning roster: {same_roster}")

    return "\n".join(lines)
