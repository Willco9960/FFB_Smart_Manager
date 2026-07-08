from dataclasses import dataclass


@dataclass
class FantasyScoringSettings:
    passing_yard: float = 0.04
    passing_touchdown: float = 4.0
    interception: float = -2.0
    rushing_yard: float = 0.1
    rushing_touchdown: float = 6.0
    reception: float = 0.0
    receiving_yard: float = 0.1
    receiving_touchdown: float = 6.0
    fumble_lost: float = -2.0


STANDARD_SCORING = FantasyScoringSettings(reception=0.0)
HALF_PPR_SCORING = FantasyScoringSettings(reception=0.5)
PPR_SCORING = FantasyScoringSettings(reception=1.0)


def get_stat_value(stats: dict[str, str], stat_name: str) -> float:
    value = stats.get(stat_name, "")

    if value == "":
        return 0.0

    return float(value)


def calculate_fantasy_points(
    stats: dict[str, str],
    scoring_settings: FantasyScoringSettings = STANDARD_SCORING,
) -> float:
    points = 0.0

    points += get_stat_value(stats, "passing_yards") * scoring_settings.passing_yard
    points += get_stat_value(stats, "passing_tds") * scoring_settings.passing_touchdown
    points += get_stat_value(stats, "interceptions") * scoring_settings.interception

    points += get_stat_value(stats, "rushing_yards") * scoring_settings.rushing_yard
    points += get_stat_value(stats, "rushing_tds") * scoring_settings.rushing_touchdown

    points += get_stat_value(stats, "receptions") * scoring_settings.reception
    points += get_stat_value(stats, "receiving_yards") * scoring_settings.receiving_yard
    points += get_stat_value(stats, "receiving_tds") * scoring_settings.receiving_touchdown

    points += get_stat_value(stats, "fumbles_lost") * scoring_settings.fumble_lost

    return round(points, 2)
