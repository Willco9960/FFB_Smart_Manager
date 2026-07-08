from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team


def test_player_creation():
    player = Player(
        name="Bijan Robinson",
        position="RB",
        team="ATL",
        projected_score=18.5,
        actual_score=21.2,
    )

    assert player.name == "Bijan Robinson"
    assert player.position == "RB"
    assert player.team == "ATL"
    assert player.projected_score == 18.5
    assert player.actual_score == 21.2
    assert player.projection_error() == 2.6999999999999993


def test_team_can_add_player():
    team = Team(name="Test Team")
    player = Player(name="CeeDee Lamb", position="WR", team="DAL", projected_score=20.0)

    team.add_player(player)

    assert team.roster_size() == 1
    assert team.roster[0].name == "CeeDee Lamb"
    assert team.projected_score() == 20.0


def test_league_can_store_teams_and_available_players():
    league = League(
        name="Test League",
        roster_rules={
            "QB": 1,
            "RB": 2,
            "WR": 2,
            "TE": 1,
            "FLEX": 1,
            "BENCH": 6,
        },
    )

    team = Team(name="Team One")
    player = Player(name="Josh Allen", position="QB", team="BUF", projected_score=24.0)

    league.add_team(team)
    league.add_available_player(player)

    assert league.team_count() == 1
    assert league.available_player_count() == 1
    assert league.roster_rules["QB"] == 1
