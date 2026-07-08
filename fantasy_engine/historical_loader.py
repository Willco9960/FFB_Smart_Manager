import csv
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_SEASON = 2021
RAW_DATA_DIR = Path("data/raw")
PLAYER_STATS_URL_TEMPLATE = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.csv"
)


def get_player_stats_url(season: int = DEFAULT_SEASON) -> str:
    return PLAYER_STATS_URL_TEMPLATE.format(season=season)


def get_player_stats_raw_path(
    season: int = DEFAULT_SEASON,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> Path:
    return raw_data_dir / f"stats_player_week_{season}.csv"


def download_player_stats(
    season: int = DEFAULT_SEASON,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> Path:
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    url = get_player_stats_url(season)
    output_path = get_player_stats_raw_path(season, raw_data_dir)

    urlretrieve(url, output_path)

    return output_path


def read_player_stats(raw_file_path: Path) -> list[dict[str, str]]:
    with raw_file_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        return list(reader)


def load_player_stats(
    season: int = DEFAULT_SEASON,
    raw_data_dir: Path = RAW_DATA_DIR,
    force_download: bool = False,
) -> list[dict[str, str]]:
    raw_file_path = get_player_stats_raw_path(season, raw_data_dir)

    if force_download or not raw_file_path.exists():
        raw_file_path = download_player_stats(season, raw_data_dir)

    return read_player_stats(raw_file_path)
