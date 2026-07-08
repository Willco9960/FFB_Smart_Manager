from dataclasses import dataclass

from fantasy_engine.team import Team


@dataclass
class TeamScore:
    team_name: str
    score: float


def score_team(team: Team) -> TeamScore:
    return TeamScore(
        team_name=team.name,
        score=round(team.actual_score(), 2),
    )


def score_teams(teams: list[Team]) -> list[TeamScore]:
    scores = []

    for team in teams:
        scores.append(score_team(team))

    return scores


def rank_team_scores(team_scores: list[TeamScore]) -> list[TeamScore]:
    return sorted(team_scores, key=lambda team_score: team_score.score, reverse=True)


def get_winner(team_scores: list[TeamScore]) -> TeamScore:
    if not team_scores:
        raise ValueError("Cannot choose a winner from an empty list of team scores.")

    ranked_scores = rank_team_scores(team_scores)

    return ranked_scores[0]


def format_ranked_team_scores(team_scores: list[TeamScore]) -> str:
    ranked_scores = rank_team_scores(team_scores)
    lines = []

    for rank, team_score in enumerate(ranked_scores, start=1):
        line = f"{rank}. {team_score.team_name}: {team_score.score}"
        lines.append(line)

    return "\n".join(lines)
