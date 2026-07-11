from dataclasses import dataclass


@dataclass(frozen=True)
class ESPNLeagueRules:
    team_count: int = 10
    regular_season_weeks: int = 14
    playoff_team_count: int = 6
    playoff_bye_team_count: int = 2


ESPN_TEN_TEAM_DEFAULT_RULES = ESPNLeagueRules()


@dataclass(frozen=True)
class ScheduledMatchup:
    week: int
    first_team_name: str
    second_team_name: str


@dataclass
class TeamStanding:
    team_name: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    points_against: float = 0.0

    def record_score(self) -> float:
        return self.wins + (self.ties * 0.5)


@dataclass(frozen=True)
class PlayoffMatchup:
    week: int
    first_seed: int
    second_seed: int


def create_round_robin_rounds(team_names: list[str]) -> list[list[tuple[str, str]]]:
    if len(team_names) % 2 != 0:
        raise ValueError("An even number of teams is required for round-robin scheduling.")

    rotating_teams = list(team_names)
    rounds = []

    for _ in range(len(team_names) - 1):
        rounds.append(
            [
                (rotating_teams[index], rotating_teams[-index - 1])
                for index in range(len(team_names) // 2)
            ]
        )
        rotating_teams = [rotating_teams[0], rotating_teams[-1], *rotating_teams[1:-1]]

    return rounds


def create_regular_season_schedule(
    team_names: list[str],
    rules: ESPNLeagueRules = ESPN_TEN_TEAM_DEFAULT_RULES,
) -> list[ScheduledMatchup]:
    if len(team_names) != rules.team_count:
        raise ValueError(f"Expected {rules.team_count} teams, received {len(team_names)}.")

    round_robin_rounds = create_round_robin_rounds(team_names)
    schedule = []

    for week in range(1, rules.regular_season_weeks + 1):
        round_matchups = round_robin_rounds[(week - 1) % len(round_robin_rounds)]

        for first_team_name, second_team_name in round_matchups:
            schedule.append(
                ScheduledMatchup(
                    week=week,
                    first_team_name=first_team_name,
                    second_team_name=second_team_name,
                )
            )

    return schedule


def initialize_standings(team_names: list[str]) -> dict[str, TeamStanding]:
    return {team_name: TeamStanding(team_name=team_name) for team_name in team_names}


def record_matchup_result(
    standings: dict[str, TeamStanding],
    matchup: ScheduledMatchup,
    first_team_score: float,
    second_team_score: float,
) -> None:
    first_team = standings[matchup.first_team_name]
    second_team = standings[matchup.second_team_name]
    first_team.points_for += first_team_score
    first_team.points_against += second_team_score
    second_team.points_for += second_team_score
    second_team.points_against += first_team_score

    if first_team_score > second_team_score:
        first_team.wins += 1
        second_team.losses += 1
    elif second_team_score > first_team_score:
        second_team.wins += 1
        first_team.losses += 1
    else:
        first_team.ties += 1
        second_team.ties += 1


def rank_standings(standings: dict[str, TeamStanding]) -> list[TeamStanding]:
    return sorted(
        standings.values(),
        key=lambda standing: (standing.record_score(), standing.points_for),
        reverse=True,
    )


def create_first_round_playoff_matchups(
    ranked_standings: list[TeamStanding],
    rules: ESPNLeagueRules = ESPN_TEN_TEAM_DEFAULT_RULES,
) -> list[PlayoffMatchup]:
    if len(ranked_standings) < rules.playoff_team_count:
        raise ValueError("Not enough ranked teams to create the playoff bracket.")

    return [
        PlayoffMatchup(week=15, first_seed=3, second_seed=6),
        PlayoffMatchup(week=15, first_seed=4, second_seed=5),
    ]
