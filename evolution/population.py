import random
from dataclasses import dataclass, field

from agents.genome_draft_agent import GenomeDraftAgent
from evolution.genome import GENOME_WEIGHT_RANGES, DraftStrategyGenome, create_random_genome
from fantasy_engine.draft import run_snake_draft
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot, score_starting_lineup
from fantasy_engine.player import Player
from fantasy_engine.scoring import TeamScore, get_winner
from fantasy_engine.team import Team


def find_team_by_name(teams: list[Team], team_name: str) -> Team:
    for team in teams:
        if team.name == team_name:
            return team

    raise ValueError(f"Could not find team named {team_name}.")


@dataclass
class EvaluatedAgent:
    genome: DraftStrategyGenome
    agent: GenomeDraftAgent
    fitness_score: float
    winning_team_name: str = ""
    winning_roster: list[Player] = field(default_factory=list)


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


def score_team_with_lineup_rules(
    team: Team,
    lineup_rules: tuple[LineupSlot, ...],
) -> TeamScore:
    try:
        lineup_score = score_starting_lineup(
            roster=team.roster,
            lineup_rules=lineup_rules,
        )
    except ValueError:
        lineup_score = 0.0

    return TeamScore(
        team_name=team.name,
        score=lineup_score,
    )


def evaluate_agent(
    agent: GenomeDraftAgent,
    league: League,
    rounds: int = 16,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
) -> EvaluatedAgent:
    test_league = clone_league_for_agent(league)

    run_snake_draft(
        league=test_league,
        rounds=rounds,
        draft_agent=agent,
    )

    team_scores = []

    for team in test_league.teams:
        team_score = score_team_with_lineup_rules(
            team=team,
            lineup_rules=lineup_rules,
        )
        team_scores.append(team_score)

    winner = get_winner(team_scores)
    winning_team = find_team_by_name(test_league.teams, winner.team_name)

    return EvaluatedAgent(
        genome=agent.genome,
        agent=agent,
        fitness_score=winner.score,
        winning_team_name=winner.team_name,
        winning_roster=list(winning_team.roster),
    )


def evaluate_population(
    agents: list[GenomeDraftAgent],
    league: League,
    rounds: int = 16,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
) -> list[EvaluatedAgent]:
    evaluated_agents = []

    for agent in agents:
        evaluated_agent = evaluate_agent(
            agent=agent,
            league=league,
            rounds=rounds,
            lineup_rules=lineup_rules,
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


def select_top_genomes(
    evaluated_agents: list[EvaluatedAgent],
    selection_count: int,
) -> list[DraftStrategyGenome]:
    ranked_agents = rank_evaluated_agents(evaluated_agents)

    selected_genomes = []

    for evaluated_agent in ranked_agents[:selection_count]:
        selected_genomes.append(evaluated_agent.genome)

    return selected_genomes


def clamp_weight(weight_name: str, value: float) -> float:
    minimum, maximum = GENOME_WEIGHT_RANGES[weight_name]

    if value < minimum:
        return minimum

    if value > maximum:
        return maximum

    return round(value, 4)


def mutate_weight(
    weight_name: str,
    current_value: float,
    rng: random.Random,
    mutation_strength: float = 0.1,
) -> float:
    mutation_amount = rng.uniform(-mutation_strength, mutation_strength)
    mutated_value = current_value + mutation_amount

    return clamp_weight(weight_name, mutated_value)


def mutate_genome(
    genome: DraftStrategyGenome,
    rng: random.Random,
    mutation_strength: float = 0.1,
) -> DraftStrategyGenome:
    return DraftStrategyGenome(
        projection_weight=mutate_weight(
            "projection_weight",
            genome.projection_weight,
            rng,
            mutation_strength,
        ),
        position_scarcity_weight=mutate_weight(
            "position_scarcity_weight",
            genome.position_scarcity_weight,
            rng,
            mutation_strength,
        ),
        adp_value_weight=mutate_weight(
            "adp_value_weight",
            genome.adp_value_weight,
            rng,
            mutation_strength,
        ),
        upside_weight=mutate_weight(
            "upside_weight",
            genome.upside_weight,
            rng,
            mutation_strength,
        ),
        floor_weight=mutate_weight(
            "floor_weight",
            genome.floor_weight,
            rng,
            mutation_strength,
        ),
        bye_week_penalty=mutate_weight(
            "bye_week_penalty",
            genome.bye_week_penalty,
            rng,
            mutation_strength,
        ),
        qb_priority=mutate_weight(
            "qb_priority",
            genome.qb_priority,
            rng,
            mutation_strength,
        ),
        rb_priority=mutate_weight(
            "rb_priority",
            genome.rb_priority,
            rng,
            mutation_strength,
        ),
        wr_priority=mutate_weight(
            "wr_priority",
            genome.wr_priority,
            rng,
            mutation_strength,
        ),
        te_priority=mutate_weight(
            "te_priority",
            genome.te_priority,
            rng,
            mutation_strength,
        ),
    )


def create_next_generation(
    selected_genomes: list[DraftStrategyGenome],
    population_size: int = 100,
    seed: int = 1,
    mutation_strength: float = 0.1,
) -> list[GenomeDraftAgent]:
    if not selected_genomes:
        raise ValueError("Cannot create next generation without selected genomes.")

    rng = random.Random(seed)
    next_generation = []

    for genome in selected_genomes:
        next_generation.append(GenomeDraftAgent(genome=genome))

    while len(next_generation) < population_size:
        parent_genome = rng.choice(selected_genomes)
        child_genome = mutate_genome(
            genome=parent_genome,
            rng=rng,
            mutation_strength=mutation_strength,
        )
        child_agent = GenomeDraftAgent(genome=child_genome)
        next_generation.append(child_agent)

    return next_generation
