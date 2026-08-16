import random
from typing import Literal

from agents.hybrid_transaction_agents import HybridTradeAgent, HybridWaiverAgent
from agents.neural_draft_agent import NeuralDraftAgent
from agents.neural_lineup_agent import NeuralLineupAgent
from agents.neural_trade_agent import NeuralTradeAgent
from agents.neural_waiver_agent import NeuralWaiverAgent
from agents.trade_agent import GenomeTradeAgent
from agents.waiver_agent import GenomeWaiverAgent
from evolution.genome import create_random_genome
from evolution.population import (
    EvaluatedAgent,
    clone_league_for_agent,
    split_population_into_leagues,
)
from fantasy_engine.draft import DraftAgent, run_snake_draft
from fantasy_engine.fitness_contract import ESPN_FITNESS_CONTRACT, FitnessContract
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot
from fantasy_engine.playoffs import simulate_espn_six_team_playoffs
from fantasy_engine.season import (
    ESPN_TEN_TEAM_DEFAULT_RULES,
    ESPNLeagueRules,
    rank_standings,
)
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from fantasy_engine.weekly_season_simulation import run_historical_regular_season
from models.transaction_value import TransactionValueNetwork
from models.weekly_projection_service import WeeklyNeuralProjectionService

WEEKLY_WIN_REWARD = 15.0
POINTS_FOR_WEIGHT = 0.05
PLAYOFF_QUALIFICATION_REWARD = 40.0
PLAYOFF_WIN_REWARD = 30.0
CHAMPIONSHIP_REWARD = 150.0
TRANSACTION_REWARD_WEIGHT = 0.25
TRANSACTION_MODES = ("neural", "genome", "hybrid", "disabled")
TransactionMode = Literal["neural", "genome", "hybrid", "disabled"]


def calculate_full_season_fitness(
    regular_season_wins: int,
    points_for: float,
    playoff_seed: int | None,
    playoff_wins: int,
    champion: bool,
    transaction_reward: float,
    lineup_efficiency: float = 0.0,
    replacement_value: float = 0.0,
    invalid_actions: int = 0,
    contract: FitnessContract = ESPN_FITNESS_CONTRACT,
) -> float:
    fitness = (regular_season_wins * contract.weekly_win_reward) + (
        points_for * contract.points_for_weight
    )

    if playoff_seed is not None:
        fitness += contract.playoff_qualification_reward

    fitness += playoff_wins * contract.playoff_win_reward

    if champion:
        fitness += contract.championship_reward

    fitness += transaction_reward * contract.transaction_reward_weight
    fitness += lineup_efficiency * contract.lineup_efficiency_weight
    fitness += replacement_value * contract.replacement_value_weight
    fitness -= invalid_actions * contract.invalid_action_penalty

    return round(fitness, 2)


def get_team_playoff_seed(
    team_name: str,
    ranked_standings,
) -> int | None:
    for seed, standing in enumerate(ranked_standings, start=1):
        if standing.team_name == team_name:
            return seed

    return None


def count_playoff_wins(team_seed: int | None, playoff_result) -> int:
    if team_seed is None:
        return 0

    return sum(
        1
        for game in playoff_result.game_results
        if game.winner_seed == team_seed
        and (game.first_seed == team_seed or game.second_seed == team_seed)
    )


def evaluate_full_season_battle_royale(
    agents: list[DraftAgent],
    league: League,
    performances: list[WeeklyPlayerPerformance],
    rounds: int = 16,
    rules: ESPNLeagueRules = ESPN_TEN_TEAM_DEFAULT_RULES,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    seed: int = 1,
    transaction_genome_fallback=None,
    projection_service: WeeklyNeuralProjectionService | None = None,
    transaction_value_model: TransactionValueNetwork | None = None,
    season: int = 0,
    transaction_mode: TransactionMode = "hybrid",
    fitness_contract: FitnessContract | None = None,
) -> list[EvaluatedAgent]:
    if len(agents) % len(league.teams) != 0:
        raise ValueError("Population size must be divisible by the league team count.")
    if transaction_mode not in TRANSACTION_MODES:
        raise ValueError(f"Unknown transaction mode: {transaction_mode}")
    if fitness_contract is not None:
        rules = fitness_contract.league_rules
        lineup_rules = fitness_contract.lineup_rules

    shuffled_agents = list(agents)
    random.Random(seed).shuffle(shuffled_agents)
    agent_groups = split_population_into_leagues(shuffled_agents, len(league.teams))
    evaluated_agents = []
    fallback_genome = transaction_genome_fallback or create_random_genome(seed=seed)

    for league_index, agent_group in enumerate(agent_groups):
        simulated_league = clone_league_for_agent(league)
        team_agents: dict[str, DraftAgent] = {}

        for team, agent in zip(simulated_league.teams, agent_group, strict=True):
            team_agents[team.name] = agent

        for agent_index, agent in enumerate(agent_group):
            reset_episode = getattr(agent, "reset_episode", None)
            if reset_episode is not None:
                reset_episode(seed + (league_index * len(agent_group)) + agent_index)

        draft_picks = run_snake_draft(
            league=simulated_league,
            rounds=rounds,
            team_agents=team_agents,
            draft_order_seed=seed + league_index,
        )
        waiver_agents = {}
        trade_agents = {}
        lineup_agents = {}

        for team, agent in zip(simulated_league.teams, agent_group, strict=True):
            if transaction_mode == "disabled":
                continue
            if transaction_mode in ("neural", "hybrid") and isinstance(agent, NeuralDraftAgent):
                neural_waiver_agent = NeuralWaiverAgent(
                    policy_network=agent.policy_network,
                    lineup_rules=lineup_rules,
                )
                neural_trade_agent = NeuralTradeAgent(
                    policy_network=agent.policy_network,
                    lineup_rules=lineup_rules,
                )
                lineup_agents[team.name] = NeuralLineupAgent(
                    policy_network=agent.policy_network,
                    lineup_rules=lineup_rules,
                )
                if transaction_mode == "hybrid":
                    fallback_waiver_agent = GenomeWaiverAgent(
                        genome=getattr(agent, "genome", None) or fallback_genome,
                        lineup_rules=lineup_rules,
                    )
                    fallback_trade_agent = GenomeTradeAgent(
                        genome=getattr(agent, "genome", None) or fallback_genome,
                        lineup_rules=lineup_rules,
                    )
                    waiver_agents[team.name] = HybridWaiverAgent(
                        neural_agent=neural_waiver_agent,
                        fallback_agent=fallback_waiver_agent,
                        value_model=transaction_value_model,
                    )
                    trade_agents[team.name] = HybridTradeAgent(
                        neural_agent=neural_trade_agent,
                        fallback_agent=fallback_trade_agent,
                        value_model=transaction_value_model,
                    )
                else:
                    waiver_agents[team.name] = neural_waiver_agent
                    trade_agents[team.name] = neural_trade_agent
            else:
                waiver_agents[team.name] = GenomeWaiverAgent(
                    genome=getattr(agent, "genome", None) or fallback_genome,
                    lineup_rules=lineup_rules,
                )
                trade_agents[team.name] = GenomeTradeAgent(
                    genome=getattr(agent, "genome", None) or fallback_genome,
                    lineup_rules=lineup_rules,
                )
        regular_season = run_historical_regular_season(
            league=simulated_league,
            performances=performances,
            rules=rules,
            lineup_rules=lineup_rules,
            waiver_agents=waiver_agents,
            trade_agents=trade_agents,
            lineup_agents=lineup_agents,
            projection_service=projection_service,
            season=season,
        )
        playoff_result = simulate_espn_six_team_playoffs(
            league=simulated_league,
            standings=regular_season.standings,
            performances=performances,
            rules=rules,
            lineup_rules=lineup_rules,
            lineup_agents=lineup_agents,
            projection_service=projection_service,
        )
        ranked_standings = rank_standings(regular_season.standings)
        transaction_rewards = {}

        for impacts in regular_season.weekly_transaction_impacts.values():
            for impact in impacts:
                transaction_rewards[impact.team_name] = (
                    transaction_rewards.get(impact.team_name, 0.0) + impact.reward
                )

        for team, agent in zip(simulated_league.teams, agent_group, strict=True):
            agent_genome = getattr(agent, "genome", None) or fallback_genome
            standing = regular_season.standings[team.name]
            playoff_seed = get_team_playoff_seed(team.name, ranked_standings)
            qualifying_playoff_seed = (
                playoff_seed
                if playoff_seed is not None and playoff_seed <= rules.playoff_team_count
                else None
            )
            playoff_wins = count_playoff_wins(playoff_seed, playoff_result)
            champion = playoff_result.champion.name == team.name
            transaction_reward = round(transaction_rewards.get(team.name, 0.0), 2)
            fitness_score = calculate_full_season_fitness(
                regular_season_wins=standing.wins,
                points_for=standing.points_for,
                playoff_seed=(
                    playoff_seed
                    if playoff_seed is not None and playoff_seed <= rules.playoff_team_count
                    else None
                ),
                playoff_wins=playoff_wins,
                champion=champion,
                transaction_reward=transaction_reward,
                contract=fitness_contract or ESPN_FITNESS_CONTRACT,
            )
            evaluated_agents.append(
                EvaluatedAgent(
                    genome=agent_genome,
                    agent=agent,
                    fitness_score=fitness_score,
                    winning_team_name=team.name,
                    winning_roster=list(team.roster),
                    winning_draft_picks=[
                        pick for pick in draft_picks if pick.team_name == team.name
                    ],
                    regular_season_wins=standing.wins,
                    points_for=round(standing.points_for, 2),
                    playoff_seed=qualifying_playoff_seed,
                    playoff_wins=playoff_wins,
                    champion=champion,
                    transaction_reward=transaction_reward,
                    playoff_rate=1.0 if qualifying_playoff_seed is not None else 0.0,
                    championship_rate=1.0 if champion else 0.0,
                )
            )

    # Keep results aligned with the caller's original agent list. This makes
    # scenario evaluation safe to distribute across worker processes without
    # relying on object identity across process boundaries.
    results_by_agent_id = {id(result.agent): result for result in evaluated_agents}
    return [results_by_agent_id[id(agent)] for agent in agents]
