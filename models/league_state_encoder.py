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
    "qb_start_need",
    "rb_start_need",
    "wr_start_need",
    "te_start_need",
    "flex_eligible_count",
    "bench_projected_points",
    "projection_floor",
    "projection_median",
    "projection_ceiling",
    "boom_probability",
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
    projection_floor: float = 0.0,
    projection_median: float = 0.0,
    projection_ceiling: float = 0.0,
    boom_probability: float = 0.0,
) -> tuple[float, ...]:
    """Create normalized state features for a team decision.

    ``current_week`` is included as a normalized clock feature.  It defaults to
    zero for draft decisions and can be supplied by weekly waiver/trade/lineup
    decisions.  No target-week or actual-score data is read here.
    """

    positions = ("QB", "RB", "WR", "TE", "K", "DST")
    starter_requirements = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    roster_features = [
        float(_count_position(team.roster, position)) / 8.0 for position in positions
    ]
    available_features = [
        float(_count_position(available_players, position)) / max(len(available_players), 1)
        for position in positions[:4]
    ]
    roster_counts = {
        position: _count_position(team.roster, position) for position in starter_requirements
    }
    starter_needs = [
        float(max(0, requirement - roster_counts[position])) / float(requirement)
        for position, requirement in starter_requirements.items()
    ]
    flex_eligible_count = sum(player.position in ("RB", "WR", "TE") for player in team.roster)
    starter_count = 1 + 2 + 2 + 1 + 1
    bench_projected_points = max(
        0.0,
        team.projected_score()
        - sum(
            sorted(
                (player.projected_score for player in team.roster),
                reverse=True,
            )[:starter_count]
        ),
    )
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
        *starter_needs,
        float(flex_eligible_count) / 16.0,
        bench_projected_points / 500.0,
        projection_floor / 500.0,
        projection_median / 500.0,
        projection_ceiling / 500.0,
        float(boom_probability),
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
