from fantasy_engine.historical_loader import (
    DEFAULT_SEASON,
    get_player_stats_raw_path,
    load_player_stats,
)


def main():
    rows = load_player_stats(DEFAULT_SEASON)
    raw_file_path = get_player_stats_raw_path(DEFAULT_SEASON)

    print(f"Loaded {len(rows)} rows for {DEFAULT_SEASON}")
    print(f"Raw data saved to: {raw_file_path}")

    if rows:
        first_row = rows[0]
        print(f"First row columns: {len(first_row)}")


if __name__ == "__main__":
    main()
