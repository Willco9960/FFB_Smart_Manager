from fantasy_engine.season import (
    ESPN_TEN_TEAM_DEFAULT_RULES,
    ScheduledMatchup,
    create_first_round_playoff_matchups,
    create_regular_season_schedule,
    initialize_standings,
    rank_standings,
    record_matchup_result,
)


def create_team_names() -> list[str]:
    return [f"Team {number}" for number in range(1, 11)]


def test_regular_season_schedule_contains_fourteen_weeks_of_matchups():
    schedule = create_regular_season_schedule(create_team_names())

    assert len(schedule) == 70
    assert {matchup.week for matchup in schedule} == set(range(1, 15))


def test_every_team_has_one_matchup_per_regular_season_week():
    schedule = create_regular_season_schedule(create_team_names())

    for week in range(1, 15):
        weekly_matchups = [matchup for matchup in schedule if matchup.week == week]
        weekly_teams = [
            team_name
            for matchup in weekly_matchups
            for team_name in (matchup.first_team_name, matchup.second_team_name)
        ]

        assert len(weekly_matchups) == 5
        assert sorted(weekly_teams) == sorted(create_team_names())


def test_record_matchup_result_updates_record_and_points():
    standings = initialize_standings(["Team 1", "Team 2"])
    matchup = ScheduledMatchup(week=1, first_team_name="Team 1", second_team_name="Team 2")

    record_matchup_result(standings, matchup, first_team_score=125.5, second_team_score=110.0)

    assert standings["Team 1"].wins == 1
    assert standings["Team 2"].losses == 1
    assert standings["Team 1"].points_for == 125.5
    assert standings["Team 2"].points_against == 125.5


def test_rank_standings_uses_points_for_as_tiebreaker():
    standings = initialize_standings(["Team 1", "Team 2"])
    standings["Team 1"].wins = 1
    standings["Team 1"].points_for = 120.0
    standings["Team 2"].wins = 1
    standings["Team 2"].points_for = 130.0

    ranked_standings = rank_standings(standings)

    assert ranked_standings[0].team_name == "Team 2"


def test_espn_six_team_playoffs_start_with_third_vs_sixth_and_fourth_vs_fifth():
    standings = initialize_standings(create_team_names())
    ranked_standings = rank_standings(standings)

    playoff_matchups = create_first_round_playoff_matchups(ranked_standings)

    assert ESPN_TEN_TEAM_DEFAULT_RULES.playoff_team_count == 6
    assert playoff_matchups[0].first_seed == 3
    assert playoff_matchups[0].second_seed == 6
    assert playoff_matchups[1].first_seed == 4
    assert playoff_matchups[1].second_seed == 5
