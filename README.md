# Fantasy Football AI Manager

A fantasy football decision engine for historical league simulation, evolutionary draft strategy, ESPN league sync, lineup optimization, custom projections, waiver recommendations, trade suggestions, and a desktop dashboard.

## Running the project locally

Activate the virtual environment, then run:

```powershell
python main.py
```

## Historical data processing

The supported walk-forward manager window is 2001–2024 with 2025 held out. The
1999–2000 files provide the two-season lookback required by the projection features.

Audit and download the required weekly files:

```powershell
python -m scripts.audit_historical_data
```

To rebuild the cleaned 2021 processed Parquet cache:

```powershell
python -m scripts.rebuild_2021_processed_season
```

Rebuild the leakage-safe draft projection checkpoint:

```powershell
python -m scripts.train_draft_projection_nn
```

Rebuild the leakage-safe weekly projection checkpoint:

```powershell
python -m scripts.train_weekly_projection_nn
```

Then train the manager across the expanded historical window:

```powershell
python -m scripts.train_manager_policy_real_seasons --population 50 --generations 40 --selection 10
```
