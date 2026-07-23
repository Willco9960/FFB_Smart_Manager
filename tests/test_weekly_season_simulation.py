from evolution.offline_replay import DecisionReplayRecord
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team
from fantasy_engine.transactions import TransactionImpact
from fantasy_engine.weekly_data import WeeklyPlayerPerformance
from fantasy_engine.weekly_season_simulation import (
    apply_transaction_rewards,
    format_final_standings,
    format_week_by_week_report,
    run_historical_regular_season,
)
from models.modular_manager_policy import create_modular_policy_features


def create_complete_team(team_number: int) -> Team:
    team_name = f"Team {team_number}"
    positions = ["QB", "RB", "RB", "WR", "WR", "TE", "WR"]
    roster = []

    for index, position in enumerate(positions, start=1):
        roster.append(
            Player(
                name=f"{team_name} {position} {index}",
                position=position,
                team="ATL",
                projected_score=20.0,
            )
        )

    return Team(name=team_name, roster=roster)


def create_performances(teams: list[Team]) -> list[WeeklyPlayerPerformance]:
    performances = []

    for week in range(1, 15):
        for team_number, team in enumerate(teams, start=1):
            for player in team.roster:
                performances.append(
                    WeeklyPlayerPerformance(
                        player_id=player.name,
                        player_name=player.name,
                        position=player.position,
                        team=player.team,
                        week=week,
                        fantasy_points=float(team_number),
                    )
                )

    return performances


def test_run_historical_regular_season_simulates_fourteen_weeks():
    teams = [create_complete_team(number) for number in range(1, 11)]
    league = League(name="Test League", teams=teams)
    performances = create_performances(teams)

    result = run_historical_regular_season(league, performances)

    assert len(result.weekly_scores) == 14
    assert len(result.weekly_standings) == 14
    assert all(len(weekly_scores) == 10 for weekly_scores in result.weekly_scores.values())
    assert all(
        standing.wins + standing.losses + standing.ties == 14
        for standing in result.standings.values()
    )


def test_final_standings_sort_highest_scoring_winning_team_first():
    teams = [create_complete_team(number) for number in range(1, 11)]
    league = League(name="Test League", teams=teams)
    performances = create_performances(teams)

    result = run_historical_regular_season(league, performances)
    standings_text = format_final_standings(result)

    assert result.ranked_standings()[0].team_name == "Team 10"
    assert "Final regular-season standings:" in standings_text


def test_week_by_week_report_includes_matchups_and_standing_snapshots():
    teams = [create_complete_team(number) for number in range(1, 11)]
    league = League(name="Test League", teams=teams)
    performances = create_performances(teams)

    result = run_historical_regular_season(league, performances)
    report = format_week_by_week_report(result)

    assert "Week 1 results:" in report
    assert "Team 1 7.00 vs Team 10 70.00" in report
    assert "Standings after Week 1:" in report


def test_transaction_rewards_accumulate_into_original_decision_across_future_weeks():
    team = Team(name="Team 1")
    incoming = Player(name="Added", position="WR", team="ATL", projected_score=10.0)
    outgoing = Player(name="Dropped", position="WR", team="ATL", projected_score=8.0)
    features = create_modular_policy_features(incoming, team, [incoming])
    records = [
        DecisionReplayRecord(
            season=2021,
            week=3,
            decision_type="waiver",
            action_key="Added",
            features=features,
            reward=2.0,
            team_name="Team 1",
        )
    ]

    updated = apply_transaction_rewards(
        records,
        [
            TransactionImpact(
                week=3,
                transaction_type="waiver",
                team_name="Team 1",
                incoming_player_names=(incoming.name,),
                outgoing_player_names=(outgoing.name,),
                incoming_points=12.0,
                outgoing_points=8.0,
                net_points=4.0,
                reward=4.0,
            ),
            TransactionImpact(
                week=4,
                transaction_type="waiver",
                team_name="Team 1",
                incoming_player_names=(incoming.name,),
                outgoing_player_names=(outgoing.name,),
                incoming_points=15.0,
                outgoing_points=5.0,
                net_points=10.0,
                reward=10.0,
            ),
        ],
    )

    assert updated[0].reward == 16.0
