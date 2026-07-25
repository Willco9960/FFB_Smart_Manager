"""Guarded transaction agents that blend learned and heuristic decisions.

The neural policy is allowed to take a transaction only when its projected
lineup value is meaningfully better than the genome fallback.  This keeps the
learned policy's upside while preventing noisy transaction heads from
overriding a safer proposal.
"""

from dataclasses import dataclass

from agents.trade_agent import TradeAgent
from agents.waiver_agent import WaiverAgent
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from fantasy_engine.transactions import TradeProposal, WaiverClaim
from models.modular_manager_policy import create_modular_policy_features
from models.transaction_value import TransactionValueNetwork


def _updated_waiver_team(team: Team, claim: WaiverClaim) -> Team:
    return Team(
        name=team.name,
        roster=[player for player in team.roster if player != claim.drop_player]
        + [claim.add_player],
    )


def _waiver_improvement(agent: WaiverAgent, team: Team, claim: WaiverClaim) -> float | None:
    scorer = getattr(agent, "get_projected_lineup_score", None)
    if scorer is None:
        return None
    baseline = scorer(team)
    updated = scorer(_updated_waiver_team(team, claim))
    if baseline is None or updated is None:
        return None
    return updated - baseline


@dataclass
class HybridWaiverAgent:
    """Choose neural waivers only when they clear the fallback by a margin."""

    neural_agent: WaiverAgent
    fallback_agent: WaiverAgent
    neural_margin: float = 0.5
    value_model: TransactionValueNetwork | None = None
    value_weight: float = 2.0
    risk_penalty: float = 0.75
    minimum_value_lower_bound: float = -0.25

    def _learned_lower_bound(
        self,
        team: Team,
        claim: WaiverClaim,
        available_players: list[Player],
        week: int,
    ) -> float:
        if self.value_model is None:
            return 0.0
        features = create_modular_policy_features(
            claim.add_player,
            team,
            available_players,
            current_week=week,
        )
        mean, uncertainty = self.value_model.score_normalized(features, "waiver")
        return mean - self.risk_penalty * uncertainty

    def choose_waiver_claim(
        self,
        team: Team,
        available_players: list[Player],
        league: League,
        week: int,
    ) -> WaiverClaim | None:
        neural_claim = self.neural_agent.choose_waiver_claim(team, available_players, league, week)
        fallback_claim = self.fallback_agent.choose_waiver_claim(
            team, available_players, league, week
        )

        if neural_claim is None:
            return fallback_claim
        if fallback_claim is None:
            if (
                self._learned_lower_bound(team, neural_claim, available_players, week)
                < self.minimum_value_lower_bound
            ):
                return None
            return neural_claim

        neural_improvement = _waiver_improvement(self.neural_agent, team, neural_claim)
        fallback_improvement = _waiver_improvement(self.fallback_agent, team, fallback_claim)
        if neural_improvement is None:
            return fallback_claim
        if fallback_improvement is None:
            return neural_claim

        neural_lower_bound = self._learned_lower_bound(
            team,
            neural_claim,
            available_players,
            week,
        )
        fallback_lower_bound = self._learned_lower_bound(
            team,
            fallback_claim,
            available_players,
            week,
        )
        neural_quality = neural_improvement + self.value_weight * neural_lower_bound
        fallback_quality = fallback_improvement + self.value_weight * fallback_lower_bound
        if (
            neural_lower_bound >= self.minimum_value_lower_bound
            and neural_quality >= fallback_quality + self.neural_margin
        ):
            return neural_claim
        return fallback_claim


def _updated_trade_teams(
    team: Team,
    opposing_team: Team,
    proposal: TradeProposal,
) -> tuple[Team, Team]:
    updated_team = Team(
        name=team.name,
        roster=[player for player in team.roster if player not in proposal.offered_players]
        + list(proposal.requested_players),
    )
    updated_opposing_team = Team(
        name=opposing_team.name,
        roster=[
            player for player in opposing_team.roster if player not in proposal.requested_players
        ]
        + list(proposal.offered_players),
    )
    return updated_team, updated_opposing_team


def _trade_value(
    agent: TradeAgent,
    team: Team,
    opposing_teams: list[Team],
    proposal: TradeProposal,
) -> tuple[float, float, float] | None:
    opposing_team = next(
        (
            candidate
            for candidate in opposing_teams
            if candidate.name == proposal.receiving_team_name
        ),
        None,
    )
    scorer = getattr(agent, "get_projected_lineup_score", None)
    if opposing_team is None or scorer is None:
        return None

    baseline_team = scorer(team)
    baseline_opposing = scorer(opposing_team)
    updated_team, updated_opposing = _updated_trade_teams(team, opposing_team, proposal)
    updated_team_score = scorer(updated_team)
    updated_opposing_score = scorer(updated_opposing)
    if (
        baseline_team is None
        or baseline_opposing is None
        or updated_team_score is None
        or updated_opposing_score is None
    ):
        return None

    team_improvement = updated_team_score - baseline_team
    opposing_improvement = updated_opposing_score - baseline_opposing
    return team_improvement, opposing_improvement, team_improvement + opposing_improvement


@dataclass
class HybridTradeAgent:
    """Choose a neural trade only when both sides and total value improve."""

    neural_agent: TradeAgent
    fallback_agent: TradeAgent
    neural_margin: float = 1.0
    value_model: TransactionValueNetwork | None = None
    value_weight: float = 2.0
    risk_penalty: float = 0.75
    minimum_value_lower_bound: float = -0.25

    def _learned_lower_bound(
        self,
        team: Team,
        opposing_team: Team,
        proposal: TradeProposal,
        week: int,
    ) -> float:
        if self.value_model is None:
            return 0.0
        requested = proposal.requested_players[0]
        offered = proposal.offered_players[0]
        requested_features = create_modular_policy_features(
            requested,
            team,
            team.roster,
            current_week=week,
        )
        offered_features = create_modular_policy_features(
            offered,
            opposing_team,
            opposing_team.roster,
            current_week=week,
        )
        requested_mean, requested_uncertainty = self.value_model.score_normalized(
            requested_features,
            "trade",
        )
        offered_mean, offered_uncertainty = self.value_model.score_normalized(
            offered_features,
            "trade",
        )
        return (requested_mean + offered_mean) / 2.0 - self.risk_penalty * (
            requested_uncertainty + offered_uncertainty
        ) / 2.0

    def choose_trade_proposal(
        self,
        team: Team,
        opposing_teams: list[Team],
        league: League,
        week: int,
    ) -> TradeProposal | None:
        neural_proposal = self.neural_agent.choose_trade_proposal(
            team, opposing_teams, league, week
        )
        fallback_proposal = self.fallback_agent.choose_trade_proposal(
            team, opposing_teams, league, week
        )

        if neural_proposal is None:
            return fallback_proposal
        if fallback_proposal is None:
            opposing_team = next(
                team for team in opposing_teams if team.name == neural_proposal.receiving_team_name
            )
            if (
                self._learned_lower_bound(team, opposing_team, neural_proposal, week)
                < self.minimum_value_lower_bound
            ):
                return None
            return neural_proposal

        neural_value = _trade_value(self.neural_agent, team, opposing_teams, neural_proposal)
        fallback_value = _trade_value(self.fallback_agent, team, opposing_teams, fallback_proposal)
        if neural_value is None:
            return fallback_proposal
        if fallback_value is None:
            return neural_proposal

        neural_team, neural_opponent, neural_total = neural_value
        if neural_team < 0.0 or neural_opponent < 0.0:
            return fallback_proposal
        neural_opposing_team = next(
            team for team in opposing_teams if team.name == neural_proposal.receiving_team_name
        )
        fallback_opposing_team = next(
            team for team in opposing_teams if team.name == fallback_proposal.receiving_team_name
        )
        neural_lower_bound = self._learned_lower_bound(
            team,
            neural_opposing_team,
            neural_proposal,
            week,
        )
        fallback_lower_bound = self._learned_lower_bound(
            team,
            fallback_opposing_team,
            fallback_proposal,
            week,
        )
        neural_quality = neural_total + self.value_weight * neural_lower_bound
        fallback_quality = fallback_value[2] + self.value_weight * fallback_lower_bound
        if (
            neural_lower_bound >= self.minimum_value_lower_bound
            and neural_quality >= fallback_quality + self.neural_margin
        ):
            return neural_proposal
        return fallback_proposal
