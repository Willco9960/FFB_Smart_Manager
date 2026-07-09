from agents.genome_draft_agent import GenomeDraftAgent
from evolution.population import (
    clone_league_for_agent,
    create_agent_population,
    evaluate_agent,
    evaluate_population,
    rank_evaluated_agents,
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
