from fantasy_engine.historical_simulation import (
    format_historical_simulation_summary,
    run_historical_draft_simulation,
)


def main():
    result = run_historical_draft_simulation()

    print(format_historical_simulation_summary(result))


if __name__ == "__main__":
    main()