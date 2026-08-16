from dataclasses import dataclass, field, replace

from agents.neural_lineup_agent import LineupAgent
from agents.trade_agent import TradeAgent
from agents.waiver_agent import WaiverAgent
from evolution.offline_replay import DecisionReplayRecord
from fantasy_engine.fitness_contract import ESPN_FITNESS_CONTRACT, FitnessContract
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot
from fantasy_engine.manager_transition import build_manager_state
from fantasy_engine.season import (
    ESPN_TEN_TEAM_DEFAULT_RULES,
    ESPNLeagueRules,
    ScheduledMatchup,
    TeamStanding,
    create_regular_season_schedule,
    initialize_standings,
    rank_standings,
)
from fantasy_engine.team import Team
from fantasy_engine.transactions import (
    Transaction,
    TransactionImpact,
    TransactionValueTracker,
    apply_trade,
    create_inverse_standings_waiver_order,
    format_transactions,
    process_waiver_claims,
)
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from fantasy_engine.weekly_projection import create_weekly_projected_roster
from fantasy_engine.weekly_simulation import simulate_historical_week
from models.modular_manager_policy import create_modular_policy_features
from models.weekly_projection_service import WeeklyNeuralProjectionService


@dataclass
class RegularSeasonSimulationResult:
    league: League
    schedule: list[ScheduledMatchup]
    standings: dict[str, TeamStanding]
    weekly_scores: dict[int, dict[str, float]]
    weekly_standings: dict[int, list[TeamStanding]]
    weekly_transactions: dict[int, list[Transaction]]
    weekly_transaction_impacts: dict[int, list[TransactionImpact]]
    decision_replay_records: list[DecisionReplayRecord] = field(default_factory=list)

    def ranked_standings(self) -> list[TeamStanding]:
        return rank_standings(self.standings)


def run_historical_regular_season(
    league: League,
    performances: list[WeeklyPlayerPerformance],
    rules: ESPNLeagueRules = ESPN_TEN_TEAM_DEFAULT_RULES,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    waiver_agents: dict[str, WaiverAgent] | None = None,
    trade_agents: dict[str, TradeAgent] | None = None,
    lineup_agents: dict[str, LineupAgent] | None = None,
    projection_service: WeeklyNeuralProjectionService | None = None,
    season: int = 0,
    fitness_contract: FitnessContract = ESPN_FITNESS_CONTRACT,
) -> RegularSeasonSimulationResult:
    team_names = [team.name for team in league.teams]
    schedule = create_regular_season_schedule(team_names, rules)
    standings = initialize_standings(team_names)
    weekly_scores = {}
    weekly_standings = {}
    weekly_transactions = {}
    weekly_transaction_impacts = {}
    transaction_tracker = TransactionValueTracker()
    decision_replay_records: list[DecisionReplayRecord] = []

    for week in range(1, rules.regular_season_weeks + 1):
        weekly_transactions[week] = run_weekly_trades(
            league=league,
            standings=standings,
            performances=performances,
            week=week,
            trade_agents=trade_agents,
            projection_service=projection_service,
            replay_records=decision_replay_records,
            season=season,
            contract_digest=fitness_contract.digest(),
        )
        weekly_transactions[week].extend(
            run_weekly_waivers(
                league=league,
                standings=standings,
                performances=performances,
                week=week,
                waiver_agents=waiver_agents,
                projection_service=projection_service,
                replay_records=decision_replay_records,
                season=season,
                contract_digest=fitness_contract.digest(),
            )
        )
        transaction_tracker.register(weekly_transactions[week])
        weekly_scores[week] = simulate_historical_week(
            teams=league.teams,
            standings=standings,
            schedule=schedule,
            performances=performances,
            week=week,
            lineup_rules=lineup_rules,
            lineup_agents=lineup_agents,
            projection_service=projection_service,
            replay_records=decision_replay_records,
            season=season,
            contract_digest=fitness_contract.digest(),
        )
        weekly_points_by_player = {}
        for performance in performances:
            if performance.week != week:
                continue
            weekly_points_by_player[(performance.player_id, performance.position)] = (
                performance.fantasy_points
            )
            # Keep the legacy key for old hand-authored players and reports.
            weekly_points_by_player[(performance.player_name, performance.position)] = (
                performance.fantasy_points
            )
        weekly_transaction_impacts[week] = transaction_tracker.evaluate_week(
            week,
            weekly_points_by_player,
        )
        decision_replay_records = apply_transaction_rewards(
            decision_replay_records,
            weekly_transaction_impacts[week],
        )
        weekly_standings[week] = [replace(standing) for standing in rank_standings(standings)]

    return RegularSeasonSimulationResult(
        league=league,
        schedule=schedule,
        standings=standings,
        weekly_scores=weekly_scores,
        weekly_standings=weekly_standings,
        weekly_transactions=weekly_transactions,
        weekly_transaction_impacts=weekly_transaction_impacts,
        decision_replay_records=decision_replay_records,
    )


def apply_transaction_rewards(
    records: list[DecisionReplayRecord],
    impacts: list[TransactionImpact],
) -> list[DecisionReplayRecord]:
    updated = list(records)
    for impact in impacts:
        incoming_key = ",".join(impact.incoming_player_names)
        matching_indices = [
            index
            for index, record in enumerate(updated)
            if (
                record.week <= impact.week
                and record.team_name == impact.team_name
                and record.decision_type == impact.transaction_type
                and record.action_key == incoming_key
                and record.executed
            )
        ]
        if matching_indices:
            # Attribute every future-week impact to the most recent matching
            # transaction.  This avoids rewarding an older add again when a
            # player is dropped and later reacquired.
            index = max(matching_indices, key=lambda item: updated[item].week)
            record = updated[index]
            updated[index] = replace(record, reward=record.reward + impact.reward)
    return updated


def run_weekly_waivers(
    league: League,
    standings: dict[str, TeamStanding],
    performances: list[WeeklyPlayerPerformance],
    week: int,
    waiver_agents: dict[str, WaiverAgent] | None,
    projection_service: WeeklyNeuralProjectionService | None = None,
    replay_records: list[DecisionReplayRecord] | None = None,
    season: int = 0,
    contract_digest: str = ESPN_FITNESS_CONTRACT.digest(),
) -> list[Transaction]:
    if waiver_agents is None:
        return []

    if projection_service is None:
        projected_available_players = create_weekly_projected_roster(
            league.available_players,
            performances,
            week,
        )
    else:
        projected_available_players = projection_service.project_roster(
            league.available_players,
            performances,
            week,
        )
    projected_players_by_key = {
        (player.name, player.position): player for player in projected_available_players
    }
    claims = []

    for team in league.teams:
        waiver_agent = waiver_agents.get(team.name)

        if waiver_agent is None:
            continue

        if projection_service is None:
            projected_roster = create_weekly_projected_roster(team.roster, performances, week)
        else:
            projected_roster = projection_service.project_roster(
                team.roster,
                performances,
                week,
            )
        projected_team = replace(team, roster=projected_roster)
        projected_claim = waiver_agent.choose_waiver_claim(
            team=projected_team,
            available_players=projected_available_players,
            league=league,
            week=week,
        )

        if projected_claim is None:
            continue

        if replay_records is not None:
            state = build_manager_state(
                projected_team,
                projected_available_players,
                season=season,
                week=week,
                contract_digest=contract_digest,
            )
            replay_records.append(
                DecisionReplayRecord(
                    season=season,
                    week=week,
                    decision_type="waiver",
                    action_key=projected_claim.add_player.name,
                    features=create_modular_policy_features(
                        projected_claim.add_player,
                        projected_team,
                        projected_available_players,
                        current_week=week,
                    ),
                    reward=0.0,
                    team_name=team.name,
                    source="historical_waiver",
                    executed=False,
                    state_digest=state.digest(),
                    contract_digest=contract_digest,
                )
            )

        original_add_player = next(
            player
            for player in league.available_players
            if (player.name, player.position)
            == (projected_claim.add_player.name, projected_claim.add_player.position)
        )
        original_drop_player = next(
            player
            for player in team.roster
            if (player.name, player.position)
            == (projected_claim.drop_player.name, projected_claim.drop_player.position)
        )

        if (
            projected_players_by_key[(original_add_player.name, original_add_player.position)]
            != projected_claim.add_player
        ):
            raise ValueError("Could not match a projected waiver player to the free-agent pool.")

        claims.append(
            replace(
                projected_claim,
                add_player=original_add_player,
                drop_player=original_drop_player,
            )
        )

    waiver_order = create_inverse_standings_waiver_order(standings)
    result = process_waiver_claims(league, claims, waiver_order)

    if replay_records is not None:
        processed_keys = {
            (claim.team_name, claim.add_player.name)
            for claim in result.processed_claims
        }
        for index, record in enumerate(replay_records):
            if (
                record.season == season
                and record.week == week
                and record.decision_type == "waiver"
            ):
                replay_records[index] = replace(
                    record,
                    executed=(record.team_name, record.action_key) in processed_keys,
                )

    return result.transactions


def run_weekly_trades(
    league: League,
    standings: dict[str, TeamStanding],
    performances: list[WeeklyPlayerPerformance],
    week: int,
    trade_agents: dict[str, TradeAgent] | None,
    projection_service: WeeklyNeuralProjectionService | None = None,
    replay_records: list[DecisionReplayRecord] | None = None,
    season: int = 0,
    contract_digest: str = ESPN_FITNESS_CONTRACT.digest(),
) -> list[Transaction]:
    if trade_agents is None:
        return []

    projected_teams = []

    for team in league.teams:
        if projection_service is None:
            projected_roster = create_weekly_projected_roster(
                team.roster,
                performances,
                week,
            )
        else:
            projected_roster = projection_service.project_roster(
                team.roster,
                performances,
                week,
            )

        projected_teams.append(replace(team, roster=projected_roster))
    projected_teams_by_name = {team.name: team for team in projected_teams}
    original_teams_by_name = {team.name: team for team in league.teams}
    projected_league = replace(league, teams=projected_teams)
    traded_team_names = set()
    transactions = []

    for team_name in create_inverse_standings_waiver_order(standings):
        if team_name in traded_team_names:
            continue

        trade_agent = trade_agents.get(team_name)

        if trade_agent is None:
            continue

        projected_team = projected_teams_by_name[team_name]
        opposing_teams = [
            team
            for team in projected_teams
            if team.name != team_name and team.name not in traded_team_names
        ]
        projected_proposal = trade_agent.choose_trade_proposal(
            team=projected_team,
            opposing_teams=opposing_teams,
            league=projected_league,
            week=week,
        )

        if projected_proposal is None:
            continue

        if replay_records is not None:
            proposing_state = build_manager_state(
                projected_team,
                projected_team.roster,
                season=season,
                week=week,
                opponent_rosters={
                    other.name: other.roster for other in projected_teams if other.name != team_name
                },
                contract_digest=contract_digest,
            )
            requested_key = ",".join(player.name for player in projected_proposal.requested_players)
            offered_key = ",".join(player.name for player in projected_proposal.offered_players)
            replay_records.extend(
                [
                    DecisionReplayRecord(
                        season=season,
                        week=week,
                        decision_type="trade",
                        action_key=requested_key,
                        features=create_modular_policy_features(
                            projected_proposal.requested_players[0],
                            projected_team,
                            projected_team.roster,
                            current_week=week,
                        ),
                        reward=0.0,
                        team_name=team_name,
                        source="historical_trade",
                        executed=False,
                        state_digest=proposing_state.digest(),
                        contract_digest=contract_digest,
                    ),
                    DecisionReplayRecord(
                        season=season,
                        week=week,
                        decision_type="trade",
                        action_key=offered_key,
                        features=create_modular_policy_features(
                            projected_proposal.offered_players[0],
                            projected_teams_by_name[projected_proposal.receiving_team_name],
                            projected_teams_by_name[projected_proposal.receiving_team_name].roster,
                            current_week=week,
                        ),
                        reward=0.0,
                        team_name=projected_proposal.receiving_team_name,
                        source="historical_trade",
                        executed=False,
                        state_digest=proposing_state.digest(),
                        contract_digest=contract_digest,
                    ),
                ]
            )

        original_proposing_team = original_teams_by_name[projected_proposal.proposing_team_name]
        original_receiving_team = original_teams_by_name[projected_proposal.receiving_team_name]
        original_offered_players = tuple(
            find_player_by_key(original_proposing_team, player.name, player.position)
            for player in projected_proposal.offered_players
        )
        original_requested_players = tuple(
            find_player_by_key(original_receiving_team, player.name, player.position)
            for player in projected_proposal.requested_players
        )
        proposal = replace(
            projected_proposal,
            offered_players=original_offered_players,
            requested_players=original_requested_players,
        )
        transactions.append(apply_trade(league, proposal))
        traded_team_names.add(proposal.proposing_team_name)
        traded_team_names.add(proposal.receiving_team_name)

        if replay_records is not None:
            executed_keys = {
                (proposal.proposing_team_name, requested_key),
                (proposal.receiving_team_name, offered_key),
            }
            for index, record in enumerate(replay_records):
                if (
                    record.season == season
                    and record.week == week
                    and record.decision_type == "trade"
                    and (record.team_name, record.action_key) in executed_keys
                ):
                    replay_records[index] = replace(record, executed=True)

    return transactions


def find_player_by_key(team: Team, player_name: str, position: str):
    for player in team.roster:
        if (player.name, player.position) == (player_name, position):
            return player

    raise ValueError(f"Could not find {player_name} ({position}) on {team.name}'s roster.")


def format_final_standings(result: RegularSeasonSimulationResult) -> str:
    lines = ["Final regular-season standings:"]

    for rank, standing in enumerate(result.ranked_standings(), start=1):
        lines.append(
            f"{rank}. {standing.team_name}: "
            f"{standing.wins}-{standing.losses}-{standing.ties}, "
            f"PF {standing.points_for:.2f}"
        )

    return "\n".join(lines)


def format_week_by_week_report(result: RegularSeasonSimulationResult) -> str:
    lines = []

    for week, weekly_scores in result.weekly_scores.items():
        lines.append(f"Week {week} transactions:")
        lines.append(format_transactions(result.weekly_transactions[week]))
        lines.append("Transaction value:")
        lines.extend(
            format_transaction_impact(impact)
            for impact in result.weekly_transaction_impacts[week]
            if impact.week == week
        )
        lines.append("")
        lines.append(f"Week {week} results:")

        for matchup in result.schedule:
            if matchup.week != week:
                continue

            lines.append(
                f"{matchup.first_team_name} {weekly_scores[matchup.first_team_name]:.2f} "
                f"vs {matchup.second_team_name} {weekly_scores[matchup.second_team_name]:.2f}"
            )

        lines.append(f"Standings after Week {week}:")

        for rank, standing in enumerate(result.weekly_standings[week], start=1):
            lines.append(
                f"{rank}. {standing.team_name}: "
                f"{standing.wins}-{standing.losses}-{standing.ties}, "
                f"PF {standing.points_for:.2f}"
            )

        lines.append("")

    return "\n".join(lines)


def format_transaction_impact(impact: TransactionImpact) -> str:
    incoming = ", ".join(impact.incoming_player_names)
    outgoing = ", ".join(impact.outgoing_player_names)

    return (
        f"{impact.team_name}: received [{incoming}] "
        f"vs gave [{outgoing}] -> "
        f"{impact.net_points:+.2f} points ({impact.outcome})"
    )
