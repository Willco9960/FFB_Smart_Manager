from dataclasses import dataclass, field

from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.season import TeamStanding
from fantasy_engine.team import Team

DEFAULT_ROSTER_LIMIT = 16


@dataclass(frozen=True)
class WaiverClaim:
    team_name: str
    add_player: Player
    drop_player: Player
    week: int


@dataclass(frozen=True)
class Transaction:
    week: int
    transaction_type: str
    team_name: str
    added_player_name: str = ""
    dropped_player_name: str = ""
    details: str = ""


@dataclass
class TransactionResult:
    processed_claims: list[WaiverClaim] = field(default_factory=list)
    rejected_claims: list[WaiverClaim] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)


def get_team_by_name(league: League, team_name: str) -> Team:
    for team in league.teams:
        if team.name == team_name:
            return team

    raise ValueError(f"Could not find team named {team_name}.")


def create_inverse_standings_waiver_order(
    standings: dict[str, TeamStanding],
) -> list[str]:
    ranked_teams = sorted(
        standings.values(),
        key=lambda standing: (standing.record_score(), standing.points_for),
    )

    return [standing.team_name for standing in ranked_teams]


def validate_waiver_claim(
    league: League,
    claim: WaiverClaim,
    roster_limit: int = DEFAULT_ROSTER_LIMIT,
) -> None:
    team = get_team_by_name(league, claim.team_name)

    if claim.add_player not in league.available_players:
        raise ValueError(f"{claim.add_player.name} is not a free agent.")

    if not team.has_player(claim.drop_player):
        raise ValueError(f"{claim.drop_player.name} is not on {team.name}'s roster.")

    if claim.add_player == claim.drop_player:
        raise ValueError("A waiver claim must add and drop different players.")

    if team.roster_size() > roster_limit:
        raise ValueError(f"{team.name} exceeds the roster limit of {roster_limit}.")


def apply_waiver_claim(
    league: League,
    claim: WaiverClaim,
    roster_limit: int = DEFAULT_ROSTER_LIMIT,
) -> Transaction:
    validate_waiver_claim(league, claim, roster_limit)
    team = get_team_by_name(league, claim.team_name)

    team.remove_player(claim.drop_player)
    league.available_players.remove(claim.add_player)
    team.add_player(claim.add_player)
    league.available_players.append(claim.drop_player)

    return Transaction(
        week=claim.week,
        transaction_type="waiver",
        team_name=team.name,
        added_player_name=claim.add_player.name,
        dropped_player_name=claim.drop_player.name,
    )


def process_waiver_claims(
    league: League,
    claims: list[WaiverClaim],
    waiver_order: list[str],
    roster_limit: int = DEFAULT_ROSTER_LIMIT,
) -> TransactionResult:
    result = TransactionResult()
    claims_by_team = {claim.team_name: claim for claim in claims}

    for team_name in waiver_order:
        claim = claims_by_team.get(team_name)

        if claim is None:
            continue

        try:
            transaction = apply_waiver_claim(league, claim, roster_limit)
        except ValueError:
            result.rejected_claims.append(claim)
            continue

        result.processed_claims.append(claim)
        result.transactions.append(transaction)

    return result


def format_transactions(transactions: list[Transaction]) -> str:
    if not transactions:
        return "No transactions."

    return "\n".join(
        f"Week {transaction.week}: {transaction.team_name} added "
        f"{transaction.added_player_name} and dropped {transaction.dropped_player_name}"
        for transaction in transactions
    )
