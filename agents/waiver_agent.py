from dataclasses import dataclass
from typing import Protocol

from agents.genome_draft_agent import GenomeDraftAgent
from evolution.genome import DraftStrategyGenome
from fantasy_engine.league import League
from fantasy_engine.lineup import (
    ESPN_OFFENSIVE_LINEUP_RULES,
    LineupSlot,
    build_best_starting_lineup,
)
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from fantasy_engine.transactions import WaiverClaim


class WaiverAgent(Protocol):
    def choose_waiver_claim(
        self,
        team: Team,
        available_players: list[Player],
        league: League,
        week: int,
    ) -> WaiverClaim | None: ...


@dataclass
class GenomeWaiverAgent:
    genome: DraftStrategyGenome
    minimum_projection_improvement: float = 0.5
    lineup_rules: tuple[LineupSlot, ...] = ESPN_OFFENSIVE_LINEUP_RULES

    def choose_waiver_claim(
        self,
        team: Team,
        available_players: list[Player],
        league: League,
        week: int,
    ) -> WaiverClaim | None:
        if not available_players or not team.roster:
            return None

        draft_agent = GenomeDraftAgent(genome=self.genome)
        add_player = max(
            available_players,
            key=lambda player: draft_agent.score_player(player, available_players),
        )
        drop_player = self.find_drop_candidate(
            team,
            add_player,
            draft_agent,
            available_players,
        )

        if drop_player is None:
            return None

        add_score = draft_agent.score_player(add_player, available_players)
        drop_score = draft_agent.score_player(drop_player, team.roster)

        if add_score - drop_score < self.minimum_projection_improvement:
            return None

        return WaiverClaim(
            team_name=team.name,
            add_player=add_player,
            drop_player=drop_player,
            week=week,
        )

    def find_drop_candidate(
        self,
        team: Team,
        add_player: Player,
        draft_agent: GenomeDraftAgent,
        available_players: list[Player],
    ) -> Player | None:
        legal_drop_candidates = []

        for player in team.roster:
            updated_roster = [
                roster_player for roster_player in team.roster if roster_player != player
            ]
            updated_roster.append(add_player)
            lineup = build_best_starting_lineup(
                roster=updated_roster,
                lineup_rules=self.lineup_rules,
                require_complete_lineup=False,
                selection_score_attribute="projected_score",
            )

            if lineup.is_complete():
                legal_drop_candidates.append(player)

        if not legal_drop_candidates:
            return None

        return min(
            legal_drop_candidates,
            key=lambda player: draft_agent.score_player(player, available_players),
        )
