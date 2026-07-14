import csv
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_SEASON = 2021
RAW_DATA_DIR = Path("data/raw")
REQUIRED_PLAYER_STATS_COLUMNS = (
    "player_id",
    "player_name",
    "position",
    "season",
    "week",
    "team",
    "opponent_team",
    "targets",
    "carries",
    "attempts",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "def_sacks",
    "def_interceptions",
    "def_tds",
    "fg_made_0_19",
    "pat_made",
)
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


def get_player_stats_header(raw_file_path: Path) -> tuple[str, ...]:
    with raw_file_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        return tuple(next(reader, []))


def validate_player_stats_schema(
    raw_file_path: Path,
    required_columns: tuple[str, ...] = REQUIRED_PLAYER_STATS_COLUMNS,
) -> tuple[str, ...]:
    header = set(get_player_stats_header(raw_file_path))
    missing_columns = [
        column for column in required_columns if column not in header and column != "team"
    ]

    if "team" in required_columns and not ({"team", "recent_team"} & header):
        missing_columns.append("team or recent_team")

    if missing_columns:
        raise ValueError(f"Historical file {raw_file_path} is missing columns: {missing_columns}")

    return tuple(sorted(header))


def load_player_stats(
    season: int = DEFAULT_SEASON,
    raw_data_dir: Path = RAW_DATA_DIR,
    force_download: bool = False,
) -> list[dict[str, str]]:
    raw_file_path = get_player_stats_raw_path(season, raw_data_dir)

    if force_download or not raw_file_path.exists():
        raw_file_path = download_player_stats(season, raw_data_dir)

    return read_player_stats(raw_file_path)
