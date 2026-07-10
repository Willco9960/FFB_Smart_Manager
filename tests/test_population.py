import random

from agents.genome_draft_agent import GenomeDraftAgent
from evolution.population import (
    clamp_weight,
    clone_league_for_agent,
    create_agent_population,
    create_next_generation,
    crossover_genomes,
    evaluate_agent,
    evaluate_population,
    evaluate_population_battle_royale,
    mutate_genome,
    mutate_weight,
    rank_evaluated_agents,
    select_top_genomes,
    split_population_into_leagues,
)
from fantasy_engine.fake_data import create_fake_player_pool
from fantasy_engine.league import League
from fantasy_engine.team import Team


def create_test_league() -> League:
    teams = [Team(name=f"Team {number}") for number in range(1, 11)]

    return League(
        name="Test League",
        teams=teams,
        available_players=create_fake_player_pool(),
    )


def test_create_agent_population_creates_requested_number_of_agents():
    agents = create_agent_population(population_size=100, seed=1)

    assert len(agents) == 100

    for agent in agents:
        assert isinstance(agent, GenomeDraftAgent)


def test_create_agent_population_is_repeatable_with_seed():
    first_agents = create_agent_population(population_size=3, seed=10)
    second_agents = create_agent_population(population_size=3, seed=10)

    assert first_agents[0].genome == second_agents[0].genome
    assert first_agents[1].genome == second_agents[1].genome
    assert first_agents[2].genome == second_agents[2].genome


def test_clone_league_for_agent_keeps_original_league_unchanged():
    league = create_test_league()

    cloned_league = clone_league_for_agent(league)

    assert cloned_league is not league
    assert cloned_league.teams is not league.teams
    assert cloned_league.available_players is not league.available_players
    assert cloned_league.team_count() == league.team_count()
    assert cloned_league.available_player_count() == league.available_player_count()


def test_evaluate_agent_assigns_fitness_score():
    league = create_test_league()
    agent = create_agent_population(population_size=1, seed=1)[0]

    evaluated_agent = evaluate_agent(
        agent=agent,
        league=league,
        rounds=16,
    )

    assert evaluated_agent.agent == agent
    assert evaluated_agent.genome == agent.genome
    assert evaluated_agent.fitness_score > 0.0
    assert evaluated_agent.winning_team_name in [team.name for team in league.teams]
    assert len(evaluated_agent.winning_roster) == 16


def test_evaluate_agent_does_not_mutate_original_league():
    league = create_test_league()
    original_available_player_count = league.available_player_count()
    agent = create_agent_population(population_size=1, seed=1)[0]

    evaluate_agent(
        agent=agent,
        league=league,
        rounds=16,
    )

    assert league.available_player_count() == original_available_player_count

    for team in league.teams:
        assert team.roster_size() == 0


def test_evaluate_population_scores_each_agent():
    league = create_test_league()
    agents = create_agent_population(population_size=5, seed=1)

    evaluated_agents = evaluate_population(
        agents=agents,
        league=league,
        rounds=16,
    )

    assert len(evaluated_agents) == 5

    for evaluated_agent in evaluated_agents:
        assert evaluated_agent.fitness_score > 0.0


def test_rank_evaluated_agents_sorts_best_to_worst():
    league = create_test_league()
    agents = create_agent_population(population_size=5, seed=1)
    evaluated_agents = evaluate_population(
        agents=agents,
        league=league,
        rounds=16,
    )

    ranked_agents = rank_evaluated_agents(evaluated_agents)

    assert ranked_agents[0].fitness_score >= ranked_agents[-1].fitness_score


def test_select_top_genomes_returns_best_genomes():
    league = create_test_league()
    agents = create_agent_population(population_size=5, seed=1)
    evaluated_agents = evaluate_population(
        agents=agents,
        league=league,
        rounds=16,
    )

    selected_genomes = select_top_genomes(
        evaluated_agents=evaluated_agents,
        selection_count=2,
    )

    ranked_agents = rank_evaluated_agents(evaluated_agents)

    assert len(selected_genomes) == 2
    assert selected_genomes[0] == ranked_agents[0].genome
    assert selected_genomes[1] == ranked_agents[1].genome


def test_clamp_weight_keeps_value_inside_genome_range():
    low_value = clamp_weight("qb_priority", -1.0)
    high_value = clamp_weight("qb_priority", 99.0)

    assert low_value == 0.0
    assert high_value == 0.35


def test_mutate_weight_changes_weight_within_safe_range():
    rng = random.Random(42)

    mutated_weight = mutate_weight(
        weight_name="rb_priority",
        current_value=0.6,
        rng=rng,
        mutation_strength=0.1,
    )

    assert 0.35 <= mutated_weight <= 1.0
    assert mutated_weight != 0.6


def test_mutate_genome_changes_at_least_one_weight():
    rng = random.Random(42)
    genome = create_agent_population(population_size=1, seed=1)[0].genome

    mutated_genome = mutate_genome(
        genome=genome,
        rng=rng,
        mutation_strength=0.1,
    )

    assert mutated_genome != genome


def test_mutate_genome_keeps_weights_inside_safe_ranges():
    rng = random.Random(42)
    genome = create_agent_population(population_size=1, seed=1)[0].genome

    mutated_genome = mutate_genome(
        genome=genome,
        rng=rng,
        mutation_strength=10.0,
    )

    assert 0.5 <= mutated_genome.projection_weight <= 1.0
    assert 0.0 <= mutated_genome.position_scarcity_weight <= 0.8
    assert 0.0 <= mutated_genome.adp_value_weight <= 1.0
    assert 0.0 <= mutated_genome.upside_weight <= 0.7
    assert 0.0 <= mutated_genome.floor_weight <= 0.7
    assert 0.0 <= mutated_genome.bye_week_penalty <= 0.3
    assert 0.0 <= mutated_genome.qb_priority <= 0.35
    assert 0.35 <= mutated_genome.rb_priority <= 1.0
    assert 0.35 <= mutated_genome.wr_priority <= 1.0
    assert 0.1 <= mutated_genome.te_priority <= 0.65


def test_create_next_generation_keeps_correct_population_size():
    league = create_test_league()
    agents = create_agent_population(population_size=5, seed=1)
    evaluated_agents = evaluate_population(
        agents=agents,
        league=league,
        rounds=16,
    )
    selected_genomes = select_top_genomes(
        evaluated_agents=evaluated_agents,
        selection_count=2,
    )

    next_generation = create_next_generation(
        selected_genomes=selected_genomes,
        population_size=100,
        seed=1,
    )

    assert len(next_generation) == 100


def test_create_next_generation_requires_selected_genomes():
    import pytest

    with pytest.raises(ValueError):
        create_next_generation(selected_genomes=[])


def test_split_population_into_ten_team_leagues():
    agents = create_agent_population(population_size=20, seed=1)

    agent_groups = split_population_into_leagues(agents, league_size=10)

    assert len(agent_groups) == 2
    assert all(len(agent_group) == 10 for agent_group in agent_groups)


def test_battle_royale_scores_every_agent_in_a_population():
    league = create_test_league()
    agents = create_agent_population(population_size=10, seed=1)

    evaluated_agents = evaluate_population_battle_royale(
        agents=agents,
        league=league,
        rounds=16,
        seed=1,
    )

    assert len(evaluated_agents) == 10
    assert {
        evaluated_agent.genome.to_json() for evaluated_agent in evaluated_agents
    } == {agent.genome.to_json() for agent in agents}
    assert all(evaluated_agent.fitness_score > 0.0 for evaluated_agent in evaluated_agents)


def test_crossover_genomes_inherits_each_weight_from_a_parent():
    first_parent = create_agent_population(population_size=1, seed=1)[0].genome
    second_parent = create_agent_population(population_size=1, seed=2)[0].genome

    child = crossover_genomes(first_parent, second_parent, random.Random(42))

    for weight_name, child_value in child.to_dict().items():
        assert child_value in {
            first_parent.to_dict()[weight_name],
            second_parent.to_dict()[weight_name],
        }
