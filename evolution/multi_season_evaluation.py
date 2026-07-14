from dataclasses import dataclass

from agents.genome_draft_agent import GenomeDraftAgent
from agents.neural_draft_agent import NeuralDraftAgent
from evolution.full_season import evaluate_full_season_battle_royale
from evolution.genome import DraftStrategyGenome
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_DEFAULT_LINEUP_RULES, LineupSlot
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from models.manager_policy_nn import ManagerPolicyNetwork
from models.weekly_projection_service import WeeklyNeuralProjectionService


@dataclass(frozen=True)
class SeasonEvaluation:
    season: int
    fitness: float
    wins: int
    points_for: float
    playoff_seed: int | None
    playoff_wins: int
    champion: bool
    transaction_reward: float
    baseline_average_fitness: float


@dataclass(frozen=True)
class MultiSeasonEvaluation:
    seasons: tuple[SeasonEvaluation, ...]

    @property
    def average_fitness(self) -> float:
        return sum(result.fitness for result in self.seasons) / len(self.seasons)

    @property
    def championship_count(self) -> int:
        return sum(result.champion for result in self.seasons)

    @property
    def playoff_count(self) -> int:
        return sum(result.playoff_seed is not None for result in self.seasons)


def evaluate_neural_manager_for_season(
    season: int,
    policy_network: ManagerPolicyNetwork,
    transaction_genome: DraftStrategyGenome,
    league: League,
    performances: list[WeeklyPlayerPerformance],
    lineup_rules: tuple[LineupSlot, ...] = ESPN_DEFAULT_LINEUP_RULES,
    seed: int = 1,
    projection_service: WeeklyNeuralProjectionService | None = None,
) -> SeasonEvaluation:
    neural_agent = NeuralDraftAgent(
        policy_network=policy_network,
        genome=transaction_genome,
    )
    baseline_agents = [
        GenomeDraftAgent(genome=transaction_genome)
        for _ in range(len(league.teams) - 1)
    ]
    evaluated_agents = evaluate_full_season_battle_royale(
        agents=[neural_agent, *baseline_agents],
        league=league,
        performances=performances,
        lineup_rules=lineup_rules,
        seed=seed,
        transaction_genome_fallback=transaction_genome,
        projection_service=projection_service,
    )
    neural_result = next(
        evaluated_agent
        for evaluated_agent in evaluated_agents
        if evaluated_agent.agent is neural_agent
    )
    baseline_fitnesses = [
        evaluated_agent.fitness_score
        for evaluated_agent in evaluated_agents
        if evaluated_agent.agent is not neural_agent
    ]

    return SeasonEvaluation(
        season=season,
        fitness=neural_result.fitness_score,
        wins=neural_result.regular_season_wins,
        points_for=neural_result.points_for,
        playoff_seed=neural_result.playoff_seed,
        playoff_wins=neural_result.playoff_wins,
        champion=neural_result.champion,
        transaction_reward=neural_result.transaction_reward,
        baseline_average_fitness=sum(baseline_fitnesses) / len(baseline_fitnesses),
    )


def format_multi_season_evaluation(result: MultiSeasonEvaluation) -> str:
    lines = ["Multi-season neural manager evaluation"]

    for season_result in result.seasons:
        lines.append(
            f"{season_result.season}: fitness {season_result.fitness:.2f}, "
            f"baseline average {season_result.baseline_average_fitness:.2f}, "
            f"record wins {season_result.wins}, "
            f"PF {season_result.points_for:.2f}, "
            f"transaction reward {season_result.transaction_reward:+.2f}, "
            f"champion {season_result.champion}"
        )

    lines.append(f"Average fitness: {result.average_fitness:.2f}")
    lines.append(f"Playoff appearances: {result.playoff_count}/{len(result.seasons)}")
    lines.append(f"Championships: {result.championship_count}/{len(result.seasons)}")

    return "\n".join(lines)
