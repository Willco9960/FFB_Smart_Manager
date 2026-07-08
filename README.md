# Fantasy Football AI Manager

A fantasy football decision engine for historical league simulation, evolutionary draft strategy, ESPN league sync, lineup optimization, custom projections, waiver recommendations, trade suggestions, and a desktop dashboard.

## Running the project locally

Activate the virtual environment, then run:

```powershell
python main.py

## Historical data processing

To rebuild the cleaned 2021 processed Parquet cache:

```powershell
python -m scripts.rebuild_2021_processed_season
