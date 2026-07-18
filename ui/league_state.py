"""Platform-neutral league summaries used by the UI shell.

The records here are demo data.  Later, ESPN, Sleeper, and NFL Fantasy
connectors can populate the same model without changing the navigation UI.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LeagueSummary:
    league_id: str
    platform: str
    league_name: str
    team_name: str
    record: str
    points_for: float
    matchup: str
    projected_points: float
    sync_status: str

    @property
    def display_name(self) -> str:
        return f"{self.platform} · {self.league_name}"


DEMO_LEAGUES = (
    LeagueSummary(
        league_id="espn-home",
        platform="ESPN",
        league_name="Home League",
        team_name="Gridiron Guardians",
        record="8-4",
        points_for=1042.6,
        matchup="vs Team 6",
        projected_points=121.4,
        sync_status="Demo data",
    ),
    LeagueSummary(
        league_id="espn-work",
        platform="ESPN",
        league_name="Work League",
        team_name="Sunday Scaries",
        record="6-6",
        points_for=987.2,
        matchup="vs Team 2",
        projected_points=108.7,
        sync_status="Demo data",
    ),
    LeagueSummary(
        league_id="sleeper-dynasty",
        platform="Sleeper",
        league_name="Dynasty League",
        team_name="The Waiver Wolves",
        record="9-3",
        points_for=1124.8,
        matchup="vs Team 4",
        projected_points=127.9,
        sync_status="Demo data",
    ),
    LeagueSummary(
        league_id="nfl-fantasy-friends",
        platform="NFL Fantasy",
        league_name="Friends League",
        team_name="Fourth Down Club",
        record="5-7",
        points_for=921.3,
        matchup="vs Team 8",
        projected_points=103.2,
        sync_status="Demo data",
    ),
)


class LeagueWorkspace:
    """Manage all connected league summaries and the active league context."""

    def __init__(self, leagues: tuple[LeagueSummary, ...] = DEMO_LEAGUES) -> None:
        self.leagues = leagues
        self._leagues_by_id = {league.league_id: league for league in leagues}
        self.active_league_id: str | None = None

    @property
    def active_league(self) -> LeagueSummary | None:
        if self.active_league_id is None:
            return None
        return self._leagues_by_id[self.active_league_id]

    @property
    def platforms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(league.platform for league in self.leagues))

    def select(self, league_id: str) -> LeagueSummary:
        if league_id not in self._leagues_by_id:
            raise ValueError(f"Unknown league: {league_id}")
        self.active_league_id = league_id
        return self._leagues_by_id[league_id]

    def clear_selection(self) -> None:
        self.active_league_id = None

    def get(self, league_id: str) -> LeagueSummary:
        if league_id not in self._leagues_by_id:
            raise ValueError(f"Unknown league: {league_id}")
        return self._leagues_by_id[league_id]
