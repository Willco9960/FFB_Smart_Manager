from fantasy_engine.historical_loader import DEFAULT_SEASON
from fantasy_engine.processed_season import (
    query_top_processed_players,
    rebuild_processed_season,
)


def main():
    parquet_path = rebuild_processed_season(DEFAULT_SEASON)
    top_players = query_top_processed_players(parquet_path, limit=10)

    print(f"Processed season saved to: {parquet_path}")
    print("Top 10 players by actual fantasy score:")

    for rank, player in enumerate(top_players, start=1):
        print(
            f"{rank}. {player['name']} "
            f"({player['position']}, {player['team']}): "
            f"{player['actual_score']}"
        )


if __name__ == "__main__":
    main()
