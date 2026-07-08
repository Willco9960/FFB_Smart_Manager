from fantasy_engine.player import Player
from fantasy_engine.scoring import (
    format_ranked_team_scores,
    get_winner,
    rank_team_scores,
    score_team,
    score_teams,
)
from fantasy_engine.team import Team


def test_score_team_uses_actual_player_scores():
    team = Team(name="Test Team")
    team.add_player(Player(name="Player 1", position="RB", team="ATL", actual_score=10.5))
    team.add_player(Player(name="Player 2", position="WR", team="DAL", actual_score=14.25))

    team_score = score_team(team)

    assert team_score.team_name == "Test Team"
    assert team_score.score == 24.75


def test_score_teams_scores_multiple_teams():
    team_one = Team(name="Team One")
    team_two = Team(name="Team Two")

    team_one.add_player(Player(name="Player 1", position="QB", team="BUF", actual_score=20.0))
    team_two.add_player(Player(name="Player 2", position="QB", team="KC", actual_score=25.0))

    scores = score_teams([team_one, team_two])

    assert len(scores) == 2
    assert scores[0].team_name == "Team One"
    assert scores[0].score == 20.0
    assert scores[1].team_name == "Team Two"
    assert scores[1].score == 25.0


def test_rank_team_scores_sorts_highest_to_lowest():
    team_one = Team(name="Team One")
    team_two = Team(name="Team Two")
    team_three = Team(name="Team Three")

    team_one.add_player(Player(name="Player 1", position="RB", team="ATL", actual_score=10.0))
    team_two.add_player(Player(name="Player 2", position="RB", team="ATL", actual_score=30.0))
    team_three.add_player(Player(name="Player 3", position="RB", team="ATL", actual_score=20.0))

    scores = score_teams([team_one, team_two, team_three])
    ranked_scores = rank_team_scores(scores)

    assert ranked_scores[0].team_name == "Team Two"
    assert ranked_scores[1].team_name == "Team Three"
    assert ranked_scores[2].team_name == "Team One"


def test_get_winner_returns_highest_scoring_team():
    team_one = Team(name="Team One")
    team_two = Team(name="Team Two")

    team_one.add_player(Player(name="Player 1", position="WR", team="DAL", actual_score=12.0))
    team_two.add_player(Player(name="Player 2", position="WR", team="MIN", actual_score=18.0))

    scores = score_teams([team_one, team_two])
    winner = get_winner(scores)

    assert winner.team_name == "Team Two"
    assert winner.score == 18.0


def test_format_ranked_team_scores_prints_scores_in_order():
    team_one = Team(name="Team One")
    team_two = Team(name="Team Two")

    team_one.add_player(Player(name="Player 1", position="TE", team="DET", actual_score=8.0))
    team_two.add_player(Player(name="Player 2", position="TE", team="SF", actual_score=16.0))

    scores = score_teams([team_one, team_two])
    formatted_scores = format_ranked_team_scores(scores)

    assert "1. Team Two: 16.0" in formatted_scores
    assert "2. Team One: 8.0" in formatted_scores
