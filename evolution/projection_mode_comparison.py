from dataclasses import dataclass

from evolution.multi_season_evaluation import SeasonEvaluation


@dataclass(frozen=True)
class ProjectionModeComparison:
    season: int
    heuristic: SeasonEvaluation
    weekly_neural: SeasonEvaluation

    @property
    def fitness_delta(self) -> float:
        return round(self.weekly_neural.fitness - self.heuristic.fitness, 2)

    @property
    def points_for_delta(self) -> float:
        return round(self.weekly_neural.points_for - self.heuristic.points_for, 2)


@dataclass(frozen=True)
class ProjectionComparison:
    seasons: tuple[ProjectionModeComparison, ...]

    @property
    def average_heuristic_fitness(self) -> float:
        return sum(item.heuristic.fitness for item in self.seasons) / len(self.seasons)

    @property
    def average_weekly_neural_fitness(self) -> float:
        return sum(item.weekly_neural.fitness for item in self.seasons) / len(self.seasons)

    @property
    def heuristic_playoff_count(self) -> int:
        return sum(item.heuristic.playoff_seed is not None for item in self.seasons)

    @property
    def weekly_neural_playoff_count(self) -> int:
        return sum(item.weekly_neural.playoff_seed is not None for item in self.seasons)

    @property
    def heuristic_championship_count(self) -> int:
        return sum(item.heuristic.champion for item in self.seasons)

    @property
    def weekly_neural_championship_count(self) -> int:
        return sum(item.weekly_neural.champion for item in self.seasons)


def format_projection_comparison(result: ProjectionComparison) -> str:
    lines = ["Heuristic versus weekly neural projection comparison"]

    for item in result.seasons:
        lines.append(
            f"{item.season}: "
            f"heuristic fitness {item.heuristic.fitness:.2f}, "
            f"weekly NN fitness {item.weekly_neural.fitness:.2f}, "
            f"delta {item.fitness_delta:+.2f}, "
            f"PF delta {item.points_for_delta:+.2f}"
        )

    lines.append(
        f"Average fitness: heuristic {result.average_heuristic_fitness:.2f}, "
        f"weekly NN {result.average_weekly_neural_fitness:.2f}"
    )
    lines.append(
        f"Playoff appearances: heuristic {result.heuristic_playoff_count}/"
        f"{len(result.seasons)}, weekly NN {result.weekly_neural_playoff_count}/"
        f"{len(result.seasons)}"
    )
    lines.append(
        f"Championships: heuristic {result.heuristic_championship_count}/"
        f"{len(result.seasons)}, weekly NN {result.weekly_neural_championship_count}/"
        f"{len(result.seasons)}"
    )

    return "\n".join(lines)
