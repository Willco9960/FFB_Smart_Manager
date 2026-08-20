from dataclasses import dataclass
from typing import Protocol

from evolution.genome import DraftStrategyGenome
from fantasy_engine.league import League
from fantasy_engine.lineup import (
    ESPN_OFFENSIVE_LINEUP_RULES,
    LineupSlot,
    build_best_starting_lineup,
)
from fantasy_engine.team import Team
from fantasy_engine.transaction_contract import stable_best
from fantasy_engine.transactions import TradeProposal


class TradeAgent(Protocol):
    def choose_trade_proposal(
        self,
        team: Team,
        opposing_teams: list[Team],
        league: League,
        week: int,
    ) -> TradeProposal | None: ...


@dataclass
class GenomeTradeAgent:
    genome: DraftStrategyGenome
    minimum_lineup_improvement: float = 0.5
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES

    def choose_trade_proposal(
        self,
        team: Team,
        opposing_teams: list[Team],
        league: League,
        week: int,
    ) -> TradeProposal | None:
        baseline_score = self.get_projected_lineup_score(team)

        if baseline_score is None:
            return None

        proposals = []

        for opposing_team in opposing_teams:
            opposing_baseline_score = self.get_projected_lineup_score(opposing_team)

            if opposing_baseline_score is None:
                continue

            for offered_player in team.roster:
                for requested_player in opposing_team.roster:
                    updated_team = Team(
                        name=team.name,
                        roster=[
                            player for player in team.roster if player != offered_player
                        ]
                        + [requested_player],
                    )
                    updated_opposing_team = Team(
                        name=opposing_team.name,
                        roster=[
                            player
                            for player in opposing_team.roster
                            if player != requested_player
                        ]
                        + [offered_player],
                    )
                    updated_team_score = self.get_projected_lineup_score(updated_team)
                    updated_opposing_score = self.get_projected_lineup_score(updated_opposing_team)

                    if updated_team_score is None or updated_opposing_score is None:
                        continue

                    team_improvement = updated_team_score - baseline_score
                    opposing_improvement = updated_opposing_score - opposing_baseline_score

                    if (
                        team_improvement < self.minimum_lineup_improvement
                        or opposing_improvement < self.minimum_lineup_improvement
                    ):
                        continue

                    combined_improvement = team_improvement + opposing_improvement

                    proposals.append(
                        (combined_improvement, TradeProposal(
                            proposing_team_name=team.name,
                            receiving_team_name=opposing_team.name,
                            offered_players=(offered_player,),
                            requested_players=(requested_player,),
                            week=week,
                        )))

        if not proposals:
            return None
        return stable_best(
            proposals,
            score=lambda item: item[0],
            tie_key=lambda item: (
                item[1].receiving_team_name,
                tuple((player.name, player.position) for player in item[1].offered_players),
                tuple((player.name, player.position) for player in item[1].requested_players),
            ),
        )[1]

    def get_projected_lineup_score(self, team: Team) -> float | None:
        lineup = build_best_starting_lineup(
            roster=team.roster,
            lineup_rules=self.lineup_rules,
            require_complete_lineup=False,
            selection_score_attribute="projected_score",
        )

        if not lineup.is_complete():
            return None

        return sum(player.projected_score for player in lineup.players)
