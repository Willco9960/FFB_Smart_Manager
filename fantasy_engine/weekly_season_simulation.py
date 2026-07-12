from dataclasses import dataclass, replace

from agents.trade_agent import TradeAgent
from agents.waiver_agent import WaiverAgent
from fantasy_engine.league import League
from fantasy_engine.lineup import ESPN_OFFENSIVE_LINEUP_RULES, LineupSlot
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


@dataclass
class RegularSeasonSimulationResult:
    league: League
    schedule: list[ScheduledMatchup]
    standings: dict[str, TeamStanding]
    weekly_scores: dict[int, dict[str, float]]
    weekly_standings: dict[int, list[TeamStanding]]
    weekly_transactions: dict[int, list[Transaction]]
    weekly_transaction_impacts: dict[int, list[TransactionImpact]]

    def ranked_standings(self) -> list[TeamStanding]:
        return rank_standings(self.standings)


def run_historical_regular_season(
    league: League,
    performances: list[WeeklyPlayerPerformance],
    rules: ESPNLeagueRules = ESPN_TEN_TEAM_DEFAULT_RULES,
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES,
    waiver_agents: dict[str, WaiverAgent] | None = None,
    trade_agents: dict[str, TradeAgent] | None = None,
) -> RegularSeasonSimulationResult:
    team_names = [team.name for team in league.teams]
    schedule = create_regular_season_schedule(team_names, rules)
    standings = initialize_standings(team_names)
    weekly_scores = {}
    weekly_standings = {}
    weekly_transactions = {}
    weekly_transaction_impacts = {}
    transaction_tracker = TransactionValueTracker()

    for week in range(1, rules.regular_season_weeks + 1):
        weekly_transactions[week] = run_weekly_trades(
            league=league,
            standings=standings,
            performances=performances,
            week=week,
            trade_agents=trade_agents,
        )
        weekly_transactions[week].extend(run_weekly_waivers(
            league=league,
            standings=standings,
            performances=performances,
            week=week,
            waiver_agents=waiver_agents,
        ))
        transaction_tracker.register(weekly_transactions[week])
        weekly_scores[week] = simulate_historical_week(
            teams=league.teams,
            standings=standings,
            schedule=schedule,
            performances=performances,
            week=week,
            lineup_rules=lineup_rules,
        )
        weekly_points_by_player = {
            (performance.player_name, performance.position): performance.fantasy_points
            for performance in performances
            if performance.week == week
        }
        weekly_transaction_impacts[week] = transaction_tracker.evaluate_week(
            week,
            weekly_points_by_player,
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
    )


def run_weekly_waivers(
    league: League,
    standings: dict[str, TeamStanding],
    performances: list[WeeklyPlayerPerformance],
    week: int,
    waiver_agents: dict[str, WaiverAgent] | None,
) -> list[Transaction]:
    if waiver_agents is None:
        return []

    projected_available_players = create_weekly_projected_roster(
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

        projected_roster = create_weekly_projected_roster(team.roster, performances, week)
        projected_team = replace(team, roster=projected_roster)
        projected_claim = waiver_agent.choose_waiver_claim(
            team=projected_team,
            available_players=projected_available_players,
            league=league,
            week=week,
        )

        if projected_claim is None:
            continue

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

    return result.transactions


def run_weekly_trades(
    league: League,
    standings: dict[str, TeamStanding],
    performances: list[WeeklyPlayerPerformance],
    week: int,
    trade_agents: dict[str, TradeAgent] | None,
) -> list[Transaction]:
    if trade_agents is None:
        return []

    projected_teams = [
        replace(
            team,
            roster=create_weekly_projected_roster(team.roster, performances, week),
        )
        for team in league.teams
    ]
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
