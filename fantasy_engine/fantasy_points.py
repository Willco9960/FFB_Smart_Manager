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
    field_goal_0_39: float = 3.0
    field_goal_40_49: float = 4.0
    field_goal_50_plus: float = 5.0
    extra_point_made: float = 1.0
    extra_point_missed: float = -1.0
    defense_sack: float = 1.0
    defense_interception: float = 2.0
    defense_fumble_recovery: float = 2.0
    defense_touchdown: float = 6.0
    defense_safety: float = 2.0


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
    position = stats.get("position", "")

    if position == "K":
        return round(
            (
                get_stat_value(stats, "fg_made_0_19")
                + get_stat_value(stats, "fg_made_20_29")
                + get_stat_value(stats, "fg_made_30_39")
            )
            * scoring_settings.field_goal_0_39
            + (
                get_stat_value(stats, "fg_made_40_49")
                * scoring_settings.field_goal_40_49
            )
            + (
                get_stat_value(stats, "fg_made_50_59")
                + get_stat_value(stats, "fg_made_60_")
            )
            * scoring_settings.field_goal_50_plus
            + get_stat_value(stats, "pat_made") * scoring_settings.extra_point_made
            + get_stat_value(stats, "pat_missed") * scoring_settings.extra_point_missed,
            2,
        )

    if position == "DST":
        return round(
            get_stat_value(stats, "def_sacks") * scoring_settings.defense_sack
            + get_stat_value(stats, "def_interceptions")
            * scoring_settings.defense_interception
            + get_stat_value(stats, "fumble_recovery_opp")
            * scoring_settings.defense_fumble_recovery
            + get_stat_value(stats, "def_tds") * scoring_settings.defense_touchdown
            + get_stat_value(stats, "def_safeties") * scoring_settings.defense_safety,
            2,
        )

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
