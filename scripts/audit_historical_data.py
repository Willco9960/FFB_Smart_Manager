import argparse
from pathlib import Path

from fantasy_engine.historical_loader import (
    RAW_DATA_DIR,
    get_player_stats_raw_path,
    load_player_stats,
    validate_player_stats_schema,
)
from fantasy_engine.historical_seasons import (
    EARLIEST_RELIABLE_SEASON,
    get_required_seasons,
    get_training_seasons,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and audit historical weekly player-stat files."
    )
    parser.add_argument("--start-season", type=int, default=EARLIEST_RELIABLE_SEASON)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--holdout-season", type=int, default=2025)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_seasons = get_training_seasons(args.start_season, args.end_season)

    if args.holdout_season <= max(training_seasons):
        raise ValueError("Holdout season must be after every training season.")

    required_seasons = tuple(
        sorted(
            set(get_required_seasons(training_seasons, lookback_seasons=2)) | {args.holdout_season}
        )
    )

    print(f"Auditing seasons: {required_seasons}")
    print(f"Raw data directory: {args.raw_data_dir}")

    for season in required_seasons:
        raw_path = get_player_stats_raw_path(season, args.raw_data_dir)
        rows = load_player_stats(season=season, raw_data_dir=args.raw_data_dir)
        validate_player_stats_schema(raw_path)

        weeks = sorted({int(row["week"]) for row in rows if row.get("week")})
        positions = sorted({row.get("position", "") for row in rows if row.get("position")})

        print(
            f"{season}: rows={len(rows):,} weeks={weeks[0]}-{weeks[-1]} "
            f"positions={','.join(positions)} path={raw_path}"
        )

    print("Historical data audit complete")


if __name__ == "__main__":
    main()
