"""Fail-closed data provenance and availability checks for training inputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from fantasy_engine.historical_loader import (
    RAW_DATA_DIR,
    get_player_stats_raw_path,
    load_player_stats,
    validate_player_stats_schema,
)

CORE_COLUMNS = (
    "player_id",
    "player_name",
    "position",
    "season",
    "week",
    "team",
)
OPTIONAL_CONTEXT_COLUMNS = (
    "injury_status",
    "status",
    "game_total",
    "spread_line",
    "implied_team_total",
    "opponent_matchup_strength",
    "qb_injury_signal",
)

# nflverse keeps a small number of player rows for which identity fields are
# blank (for example, stat-only rows).  A 99% threshold remains fail-closed for
# corrupt/truncated files while accepting that documented source behavior.
DEFAULT_MINIMUM_CORE_COVERAGE = 0.99


@dataclass(frozen=True)
class DataAvailabilityManifest:
    season: int
    source_path: str
    row_count: int
    columns: tuple[str, ...]
    coverage: tuple[tuple[str, float], ...]
    missing_optional_columns: tuple[str, ...]

    def coverage_for(self, column: str) -> float:
        return dict(self.coverage).get(column, 0.0)

    def require(
        self,
        columns: Iterable[str] = CORE_COLUMNS,
        minimum_coverage: float = DEFAULT_MINIMUM_CORE_COVERAGE,
    ) -> None:
        if not 0.0 < minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be in (0, 1].")
        missing = [
            column
            for column in columns
            if column not in self.columns
            and not (column == "team" and "recent_team" in self.columns)
        ]
        low_coverage = [
            column
            for column in columns
            if column in self.columns and self.coverage_for(column) < minimum_coverage
        ]
        if missing or low_coverage:
            raise ValueError(
                f"Season {self.season} data is not training-ready: "
                f"missing={missing}, low_coverage={low_coverage}."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "season": self.season,
            "source_path": self.source_path,
            "row_count": self.row_count,
            "columns": list(self.columns),
            "coverage": dict(self.coverage),
            "missing_optional_columns": list(self.missing_optional_columns),
        }


def inspect_rows(
    season: int,
    rows: list[dict[str, str]],
    source_path: Path | str,
) -> DataAvailabilityManifest:
    if not rows:
        raise ValueError(f"Season {season} data is empty: {source_path}")
    columns = tuple(sorted({key for row in rows for key in row}))
    coverage = tuple(
        sorted(
            (
                column,
                sum(bool(row.get(column, "")) for row in rows) / len(rows),
            )
            for column in columns
        )
    )
    return DataAvailabilityManifest(
        season=season,
        source_path=str(source_path),
        row_count=len(rows),
        columns=columns,
        coverage=coverage,
        missing_optional_columns=tuple(
            column for column in OPTIONAL_CONTEXT_COLUMNS if column not in columns
        ),
    )


def validate_training_seasons(
    seasons: Iterable[int],
    *,
    raw_data_dir: Path = RAW_DATA_DIR,
    minimum_core_coverage: float = DEFAULT_MINIMUM_CORE_COVERAGE,
) -> tuple[DataAvailabilityManifest, ...]:
    manifests = []
    for season in seasons:
        path = get_player_stats_raw_path(season, raw_data_dir)
        rows = load_player_stats(season=season, raw_data_dir=raw_data_dir)
        validate_player_stats_schema(path)
        manifest = inspect_rows(season, rows, path)
        manifest.require(minimum_coverage=minimum_core_coverage)
        manifests.append(manifest)
    return tuple(manifests)
