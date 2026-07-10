import random
from dataclasses import dataclass, field

from agents.baseline_agents import create_baseline_opponents
from agents.genome_draft_agent import GenomeDraftAgent
from evolution.genome import GENOME_WEIGHT_RANGES, DraftStrategyGenome, create_random_genome
from fantasy_engine.draft import DraftAgent, run_snake_draft
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot, score_starting_lineup
from fantasy_engine.player import Player
from fantasy_engine.scoring import TeamScore
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
    seed: int = 1,
) -> EvaluatedAgent:
    test_league = clone_league_for_agent(league)
    rng = random.Random(seed)
    target_team = rng.choice(test_league.teams)
    opponents = create_baseline_opponents(
        opponent_count=len(test_league.teams) - 1,
        seed=seed,
    )
    team_agents: dict[str, DraftAgent] = {target_team.name: agent}

    for team, opponent in zip(
        (team for team in test_league.teams if team is not target_team),
        opponents,
        strict=True,
    ):
        team_agents[team.name] = opponent

    run_snake_draft(
        league=test_league,
        rounds=rounds,
        team_agents=team_agents,
    )
    target_team_score = score_team_with_lineup_rules(
        team=target_team,
        lineup_rules=lineup_rules,
    )

    return EvaluatedAgent(
        genome=agent.genome,
        agent=agent,
        fitness_score=target_team_score.score,
        winning_team_name=target_team.name,
        winning_roster=list(target_team.roster),
    )


def evaluate_population(
    agents: list[GenomeDraftAgent],
    league: League,
    rounds: int = 16,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    seed: int = 1,
) -> list[EvaluatedAgent]:
    evaluated_agents = []

    for index, agent in enumerate(agents):
        evaluated_agent = evaluate_agent(
            agent=agent,
            league=league,
            rounds=rounds,
            lineup_rules=lineup_rules,
            seed=seed + index,
        )
        evaluated_agents.append(evaluated_agent)

    return evaluated_agents


def split_population_into_leagues(
    agents: list[GenomeDraftAgent],
    league_size: int,
) -> list[list[GenomeDraftAgent]]:
    if league_size <= 0:
        raise ValueError("League size must be positive.")

    if len(agents) % league_size != 0:
        raise ValueError("Population size must be divisible by the league size.")

    return [agents[index : index + league_size] for index in range(0, len(agents), league_size)]


def evaluate_population_battle_royale(
    agents: list[GenomeDraftAgent],
    league: League,
    rounds: int = 16,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    seed: int = 1,
) -> list[EvaluatedAgent]:
    league_size = len(league.teams)
    shuffled_agents = list(agents)
    random.Random(seed).shuffle(shuffled_agents)
    agent_groups = split_population_into_leagues(shuffled_agents, league_size)
    evaluated_agents = []

    for agent_group in agent_groups:
        test_league = clone_league_for_agent(league)
        team_agents: dict[str, DraftAgent] = {}

        for team, agent in zip(test_league.teams, agent_group, strict=True):
            team_agents[team.name] = agent

        run_snake_draft(
            league=test_league,
            rounds=rounds,
            team_agents=team_agents,
        )

        for team, agent in zip(test_league.teams, agent_group, strict=True):
            team_score = score_team_with_lineup_rules(team, lineup_rules)
            evaluated_agents.append(
                EvaluatedAgent(
                    genome=agent.genome,
                    agent=agent,
                    fitness_score=team_score.score,
                    winning_team_name=team.name,
                    winning_roster=list(team.roster),
                )
            )

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


def crossover_genomes(
    first_parent: DraftStrategyGenome,
    second_parent: DraftStrategyGenome,
    rng: random.Random,
) -> DraftStrategyGenome:
    first_parent_data = first_parent.to_dict()
    second_parent_data = second_parent.to_dict()
    child_data = {}

    for weight_name in first_parent_data:
        child_data[weight_name] = rng.choice(
            [first_parent_data[weight_name], second_parent_data[weight_name]]
        )

    return DraftStrategyGenome.from_dict(child_data)


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
        first_parent = rng.choice(selected_genomes)
        second_parent = rng.choice(selected_genomes)
        child_genome = crossover_genomes(first_parent, second_parent, rng)
        child_genome = mutate_genome(
            genome=child_genome,
            rng=rng,
            mutation_strength=mutation_strength,
        )
        child_agent = GenomeDraftAgent(genome=child_genome)
        next_generation.append(child_agent)

    return next_generation
