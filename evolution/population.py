from dataclasses import dataclass

from agents.genome_draft_agent import GenomeDraftAgent
from evolution.genome import DraftStrategyGenome, create_random_genome
from fantasy_engine.draft import run_snake_draft
from fantasy_engine.league import League
from fantasy_engine.scoring import get_winner, score_teams
from fantasy_engine.team import Team


@dataclass
class EvaluatedAgent:
    genome: DraftStrategyGenome
    agent: GenomeDraftAgent
    fitness_score: float


def create_agent_population(
    population_size: int = 100,
    seed: int = 1,
) -> list[GenomeDraftAgent]:
    agents = []

    for index in range(population_size):
        genome = create_random_genome(seed=seed + index)
        agent = GenomeDraftAgent(genome=genome)
        agents.append(agent)

    return agents


def clone_league_for_agent(league: League) -> League:
    teams = []

    for team in league.teams:
        teams.append(Team(name=team.name))

    return League(
        name=league.name,
        teams=teams,
        available_players=list(league.available_players),
        roster_rules=dict(league.roster_rules),
    )


def evaluate_agent(
    agent: GenomeDraftAgent,
    league: League,
    rounds: int = 16,
) -> EvaluatedAgent:
    test_league = clone_league_for_agent(league)

    run_snake_draft(
        league=test_league,
        rounds=rounds,
        draft_agent=agent,
    )

    team_scores = score_teams(test_league.teams)
    winner = get_winner(team_scores)

    return EvaluatedAgent(
        genome=agent.genome,
        agent=agent,
        fitness_score=winner.score,
    )


def evaluate_population(
    agents: list[GenomeDraftAgent],
    league: League,
    rounds: int = 16,
) -> list[EvaluatedAgent]:
    evaluated_agents = []

    for agent in agents:
        evaluated_agent = evaluate_agent(
            agent=agent,
            league=league,
            rounds=rounds,
        )
        evaluated_agents.append(evaluated_agent)

    return evaluated_agents


def rank_evaluated_agents(
    evaluated_agents: list[EvaluatedAgent],
) -> list[EvaluatedAgent]:
    return sorted(
        evaluated_agents,
        key=lambda evaluated_agent: evaluated_agent.fitness_score,
        reverse=True,
    )
