from agents.genome_draft_agent import GenomeDraftAgent
from agents.waiver_agent import GenomeWaiverAgent
from evolution.population import create_agent_population
from fantasy_engine.draft import run_snake_draft
from fantasy_engine.league import League
from fantasy_engine.leakage_safe_player_pool import load_leakage_safe_player_pool
from fantasy_engine.playoffs import format_playoff_result, simulate_espn_six_team_playoffs
from fantasy_engine.team import Team
from fantasy_engine.weekly_data import load_weekly_performances
from fantasy_engine.weekly_season_simulation import (
    format_final_standings,
    format_week_by_week_report,
    run_historical_regular_season,
)


def create_2021_historical_league() -> League:
    teams = [Team(name=f"Weekly Team {number}") for number in range(1, 11)]
    players = load_leakage_safe_player_pool(
        projection_season=2020,
        actual_season=2021,
    )[:250]

    return League(
        name="2021 Weekly Historical League",
        teams=teams,
        available_players=players,
    )


def run_draft(league: League) -> dict[str, GenomeDraftAgent]:
    agents = create_agent_population(population_size=len(league.teams), seed=2021)
    team_agents: dict[str, GenomeDraftAgent] = {}

    for team, agent in zip(league.teams, agents, strict=True):
        team_agents[team.name] = agent

    run_snake_draft(league=league, rounds=16, team_agents=team_agents)

    return team_agents


def main():
    league = create_2021_historical_league()
    draft_agents = run_draft(league)
    waiver_agents = {
        team_name: GenomeWaiverAgent(genome=draft_agent.genome)
        for team_name, draft_agent in draft_agents.items()
    }
    performances = load_weekly_performances(2021)
    result = run_historical_regular_season(
        league,
        performances,
        waiver_agents=waiver_agents,
    )
    playoff_result = simulate_espn_six_team_playoffs(league, result.standings, performances)

    print("2021 historical weekly regular-season simulation complete")
    print(
        "Lineups adapt from prior-week performance; "
        "historical weekly scores are applied afterward."
    )
    print("Waiver claims process before each week's lineup decisions.")
    print("")
    print(format_week_by_week_report(result))
    print("")
    print(format_final_standings(result))
    print("")
    print(format_playoff_result(playoff_result))


if __name__ == "__main__":
    main()
