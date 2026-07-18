"""Shared, leakage-safe representations of the current fantasy-league state.

The encoder deliberately uses only information supplied by the caller.  It does
not inspect actual scores or future weeks, which makes it safe to use during a
historical simulation and in the live assistant later.
"""

from fantasy_engine.player import Player
from fantasy_engine.team import Team

LEAGUE_STATE_FEATURE_NAMES = (
    "roster_size",
    "roster_qb_count",
    "roster_rb_count",
    "roster_wr_count",
    "roster_te_count",
    "roster_k_count",
    "roster_dst_count",
    "available_player_count",
    "available_qb_count",
    "available_rb_count",
    "available_wr_count",
    "available_te_count",
    "team_projected_points",
    "best_available_projection",
    "current_week",
    "projection_uncertainty",
    "opponent_strength",
    "standing_win_rate",
    "playoff_probability",
)


def _count_position(players: list[Player], position: str) -> int:
    return sum(player.position == position for player in players)


def create_league_state_features(
    team: Team,
    available_players: list[Player],
    current_week: int = 0,
    regular_season_weeks: int = 14,
    projection_uncertainty: float = 0.0,
    opponent_strength: float = 0.0,
    standing_win_rate: float = 0.0,
    playoff_probability: float = 0.0,
) -> tuple[float, ...]:
    """Create normalized state features for a team decision.

    ``current_week`` is included as a normalized clock feature.  It defaults to
    zero for draft decisions and can be supplied by weekly waiver/trade/lineup
    decisions.  No target-week or actual-score data is read here.
    """

    positions = ("QB", "RB", "WR", "TE", "K", "DST")
    roster_features = [
        float(_count_position(team.roster, position)) / 8.0 for position in positions
    ]
    available_features = [
        float(_count_position(available_players, position)) / max(len(available_players), 1)
        for position in positions[:4]
    ]
    return (
        float(len(team.roster)) / 16.0,
        *roster_features,
        float(len(available_players)) / 250.0,
        *available_features,
        float(team.projected_score()) / 1000.0,
        max((player.projected_score for player in available_players), default=0.0) / 500.0,
        float(current_week) / max(float(regular_season_weeks), 1.0),
        float(projection_uncertainty) / 100.0,
        float(opponent_strength) / 100.0,
        float(standing_win_rate),
        float(playoff_probability),
    )


def create_player_set_summary(players: list[Player]) -> tuple[float, ...]:
    """Summarize a variable-sized player set for diagnostics and value heads."""

    if not players:
        return (0.0, 0.0, 0.0, 0.0)

    projections = [player.projected_score for player in players]
    return (
        float(len(players)),
        sum(projections) / len(projections),
        max(projections),
        min(projections),
    )
