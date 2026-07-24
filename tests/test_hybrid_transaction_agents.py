from dataclasses import dataclass

from agents.hybrid_transaction_agents import HybridTradeAgent, HybridWaiverAgent
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from fantasy_engine.transactions import TradeProposal, WaiverClaim


def player(name: str, position: str, projection: float) -> Player:
    return Player(name=name, position=position, team="TEST", projected_score=projection)


def complete_team(name: str) -> Team:
    return Team(
        name=name,
        roster=[
            player(f"{name} QB", "QB", 20),
            player(f"{name} RB1", "RB", 18),
            player(f"{name} RB2", "RB", 17),
            player(f"{name} WR1", "WR", 16),
            player(f"{name} WR2", "WR", 15),
            player(f"{name} TE", "TE", 12),
        ],
    )


@dataclass
class FixedWaiverAgent:
    claim: WaiverClaim | None

    def choose_waiver_claim(self, team, available_players, league, week):
        return self.claim

    def get_projected_lineup_score(self, team):
        return sum(player.projected_score for player in team.roster)


@dataclass
class FixedTradeAgent:
    proposal: TradeProposal | None

    def choose_trade_proposal(self, team, opposing_teams, league, week):
        return self.proposal

    def get_projected_lineup_score(self, team):
        return sum(player.projected_score for player in team.roster)


def test_hybrid_waiver_prefers_neural_only_when_margin_is_cleared():
    team = complete_team("Team")
    weak_free_agent = player("Fallback Add", "WR", 20)
    strong_free_agent = player("Neural Add", "WR", 24)
    drop_player = team.roster[-1]
    fallback_claim = WaiverClaim(team.name, weak_free_agent, drop_player, 2)
    neural_claim = WaiverClaim(team.name, strong_free_agent, drop_player, 2)
    league = League(name="Test", teams=[team])

    result = HybridWaiverAgent(
        neural_agent=FixedWaiverAgent(neural_claim),
        fallback_agent=FixedWaiverAgent(fallback_claim),
        neural_margin=0.5,
    ).choose_waiver_claim(team, [weak_free_agent, strong_free_agent], league, 2)

    assert result == neural_claim


def test_hybrid_waiver_falls_back_when_neural_margin_is_not_cleared():
    team = complete_team("Team")
    fallback_player = player("Fallback Add", "WR", 25)
    neural_player = player("Neural Add", "WR", 24)
    drop_player = team.roster[-1]
    fallback_claim = WaiverClaim(team.name, fallback_player, drop_player, 2)
    neural_claim = WaiverClaim(team.name, neural_player, drop_player, 2)
    league = League(name="Test", teams=[team])

    result = HybridWaiverAgent(
        neural_agent=FixedWaiverAgent(neural_claim),
        fallback_agent=FixedWaiverAgent(fallback_claim),
        neural_margin=0.5,
    ).choose_waiver_claim(team, [fallback_player, neural_player], league, 2)

    assert result == fallback_claim


def test_hybrid_trade_rejects_neural_trade_that_does_not_help_both_sides():
    team = complete_team("Team")
    opposing = complete_team("Opponent")
    offered = team.roster[-1]
    requested = opposing.roster[-1]
    fallback_proposal = TradeProposal(team.name, opposing.name, (offered,), (requested,), 3)
    neural_proposal = TradeProposal(
        team.name, opposing.name, (team.roster[0],), (opposing.roster[0],), 3
    )
    league = League(name="Test", teams=[team, opposing])

    result = HybridTradeAgent(
        neural_agent=FixedTradeAgent(neural_proposal),
        fallback_agent=FixedTradeAgent(fallback_proposal),
    ).choose_trade_proposal(team, [opposing], league, 3)

    assert result == fallback_proposal
