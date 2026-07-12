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
class TradeProposal:
    proposing_team_name: str
    receiving_team_name: str
    offered_players: tuple[Player, ...]
    requested_players: tuple[Player, ...]
    week: int


@dataclass(frozen=True)
class Transaction:
    week: int
    transaction_type: str
    team_name: str
    added_player_name: str = ""
    dropped_player_name: str = ""
    details: str = ""
    added_player: Player | None = None
    dropped_player: Player | None = None
    offered_players: tuple[Player, ...] = ()
    received_players: tuple[Player, ...] = ()
    counterparty_team_name: str = ""


@dataclass(frozen=True)
class TransactionImpact:
    week: int
    transaction_type: str
    team_name: str
    incoming_player_names: tuple[str, ...]
    outgoing_player_names: tuple[str, ...]
    incoming_points: float
    outgoing_points: float
    net_points: float
    reward: float

    @property
    def outcome(self) -> str:
        if self.net_points > 0:
            return "positive"

        if self.net_points < 0:
            return "negative"

        return "neutral"


@dataclass
class TransactionResult:
    processed_claims: list[WaiverClaim] = field(default_factory=list)
    rejected_claims: list[WaiverClaim] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)


@dataclass
class TransactionValueTracker:
    active_transactions: list[Transaction] = field(default_factory=list)
    impacts_by_week: dict[int, list[TransactionImpact]] = field(default_factory=dict)

    def register(self, transactions: list[Transaction]) -> None:
        self.active_transactions.extend(transactions)

    def evaluate_week(
        self,
        week: int,
        weekly_points_by_player: dict[tuple[str, str], float],
    ) -> list[TransactionImpact]:
        impacts = []

        for transaction in self.active_transactions:
            if transaction.week > week:
                continue

            team_sides = get_transaction_sides(transaction)

            for team_name, incoming_players, outgoing_players in team_sides:
                incoming_points = round(
                    sum(
                        weekly_points_by_player.get(
                            (player.name, player.position),
                            0.0,
                        )
                        for player in incoming_players
                    ),
                    2,
                )
                outgoing_points = round(
                    sum(
                        weekly_points_by_player.get(
                            (player.name, player.position),
                            0.0,
                        )
                        for player in outgoing_players
                    ),
                    2,
                )
                net_points = round(incoming_points - outgoing_points, 2)
                impacts.append(
                    TransactionImpact(
                        week=week,
                        transaction_type=transaction.transaction_type,
                        team_name=team_name,
                        incoming_player_names=tuple(
                            player.name for player in incoming_players
                        ),
                        outgoing_player_names=tuple(
                            player.name for player in outgoing_players
                        ),
                        incoming_points=incoming_points,
                        outgoing_points=outgoing_points,
                        net_points=net_points,
                        reward=net_points,
                    )
                )

        self.impacts_by_week[week] = impacts
        return impacts


def get_transaction_sides(
    transaction: Transaction,
) -> list[tuple[str, tuple[Player, ...], tuple[Player, ...]]]:
    if transaction.transaction_type == "waiver":
        if transaction.added_player is None or transaction.dropped_player is None:
            return []

        return [
            (
                transaction.team_name,
                (transaction.added_player,),
                (transaction.dropped_player,),
            )
        ]

    return [
        (
            transaction.team_name,
            transaction.received_players,
            transaction.offered_players,
        ),
        (
            transaction.counterparty_team_name,
            transaction.offered_players,
            transaction.received_players,
        ),
    ]


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
        added_player=claim.add_player,
        dropped_player=claim.drop_player,
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
        format_transaction(transaction)
        for transaction in transactions
    )


def validate_trade_proposal(league: League, proposal: TradeProposal) -> None:
    if proposal.proposing_team_name == proposal.receiving_team_name:
        raise ValueError("A team cannot trade with itself.")

    if not proposal.offered_players or not proposal.requested_players:
        raise ValueError("A trade must include players from both teams.")

    proposing_team = get_team_by_name(league, proposal.proposing_team_name)
    receiving_team = get_team_by_name(league, proposal.receiving_team_name)

    offered_player_keys = {(player.name, player.position) for player in proposal.offered_players}
    requested_player_keys = {
        (player.name, player.position) for player in proposal.requested_players
    }

    if len(offered_player_keys) != len(proposal.offered_players):
        raise ValueError("A trade cannot offer the same player more than once.")

    if len(requested_player_keys) != len(proposal.requested_players):
        raise ValueError("A trade cannot request the same player more than once.")

    if not all(proposing_team.has_player(player) for player in proposal.offered_players):
        raise ValueError("The proposing team does not own every offered player.")

    if not all(receiving_team.has_player(player) for player in proposal.requested_players):
        raise ValueError("The receiving team does not own every requested player.")


def apply_trade(league: League, proposal: TradeProposal) -> Transaction:
    validate_trade_proposal(league, proposal)
    proposing_team = get_team_by_name(league, proposal.proposing_team_name)
    receiving_team = get_team_by_name(league, proposal.receiving_team_name)

    for player in proposal.offered_players:
        proposing_team.remove_player(player)
        receiving_team.add_player(player)

    for player in proposal.requested_players:
        receiving_team.remove_player(player)
        proposing_team.add_player(player)

    offered_names = ", ".join(player.name for player in proposal.offered_players)
    requested_names = ", ".join(player.name for player in proposal.requested_players)

    return Transaction(
        week=proposal.week,
        transaction_type="trade",
        team_name=proposing_team.name,
        details=(
            f"{proposing_team.name} sent {offered_names} to {receiving_team.name} "
            f"for {requested_names}"
        ),
        offered_players=proposal.offered_players,
        received_players=proposal.requested_players,
        counterparty_team_name=receiving_team.name,
    )


def format_transaction(transaction: Transaction) -> str:
    if transaction.transaction_type == "trade":
        return f"Week {transaction.week}: TRADE - {transaction.details}"

    return (
        f"Week {transaction.week}: {transaction.team_name} added "
        f"{transaction.added_player_name} and dropped {transaction.dropped_player_name}"
    )
