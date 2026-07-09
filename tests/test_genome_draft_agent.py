import pytest

from agents.genome_draft_agent import GenomeDraftAgent
from evolution.genome import DraftStrategyGenome
from fantasy_engine.draft import run_snake_draft
from fantasy_engine.fake_data import create_fake_player_pool
from fantasy_engine.league import League
from fantasy_engine.player import Player
from fantasy_engine.team import Team


def create_test_genome() -> DraftStrategyGenome:
    return DraftStrategyGenome(
        projection_weight=1.0,
        position_scarcity_weight=0.0,
        adp_value_weight=0.0,
        upside_weight=0.0,
        floor_weight=0.0,
        bye_week_penalty=0.0,
        qb_priority=0.0,
        rb_priority=0.0,
        wr_priority=0.0,
        te_priority=0.0,
    )


def test_genome_draft_agent_raises_error_for_empty_player_pool():
    agent = GenomeDraftAgent(genome=create_test_genome())
    team = Team(name="Test Team")
    league = League(name="Test League", teams=[team], available_players=[])

    with pytest.raises(ValueError):
        agent.choose_player([], team, league)


def test_genome_draft_agent_chooses_highest_projected_player():
    lower_player = Player(
        name="Lower Player",
        position="RB",
        team="ATL",
        projected_score=10.0,
    )
    higher_player = Player(
        name="Higher Player",
        position="RB",
        team="BUF",
        projected_score=20.0,
    )
    available_players = [lower_player, higher_player]

    agent = GenomeDraftAgent(genome=create_test_genome())
    team = Team(name="Test Team")
    league = League(
        name="Test League",
        teams=[team],
        available_players=available_players,
    )

    selected_player = agent.choose_player(available_players, team, league)

    assert selected_player == higher_player


def test_genome_draft_agent_uses_position_priority():
    rb_player = Player(
        name="RB Player",
        position="RB",
        team="ATL",
        projected_score=10.0,
    )
    wr_player = Player(
        name="WR Player",
        position="WR",
        team="BUF",
        projected_score=10.0,
    )
    genome = DraftStrategyGenome(
        projection_weight=0.0,
        position_scarcity_weight=0.0,
        adp_value_weight=0.0,
        upside_weight=0.0,
        floor_weight=0.0,
        bye_week_penalty=0.0,
        qb_priority=0.0,
        rb_priority=1.0,
        wr_priority=0.0,
        te_priority=0.0,
    )
    available_players = [wr_player, rb_player]

    agent = GenomeDraftAgent(genome=genome)
    team = Team(name="Test Team")
    league = League(
        name="Test League",
        teams=[team],
        available_players=available_players,
    )

    selected_player = agent.choose_player(available_players, team, league)

    assert selected_player == rb_player


def test_genome_draft_agent_scores_position_scarcity():
    scarce_te = Player(
        name="Scarce TE",
        position="TE",
        team="ATL",
        projected_score=10.0,
    )
    rb_one = Player(
        name="RB One",
        position="RB",
        team="BUF",
        projected_score=10.0,
    )
    rb_two = Player(
        name="RB Two",
        position="RB",
        team="CIN",
        projected_score=10.0,
    )
    genome = DraftStrategyGenome(
        projection_weight=0.0,
        position_scarcity_weight=1.0,
        adp_value_weight=0.0,
        upside_weight=0.0,
        floor_weight=0.0,
        bye_week_penalty=0.0,
        qb_priority=0.0,
        rb_priority=0.0,
        wr_priority=0.0,
        te_priority=0.0,
    )
    available_players = [scarce_te, rb_one, rb_two]

    agent = GenomeDraftAgent(genome=genome)

    te_score = agent.score_player(scarce_te, available_players)
    rb_score = agent.score_player(rb_one, available_players)

    assert te_score > rb_score


def test_genome_draft_agent_scores_position_value():
    strong_rb = Player(
        name="Strong RB",
        position="RB",
        team="ATL",
        projected_score=20.0,
    )
    weak_rb = Player(
        name="Weak RB",
        position="RB",
        team="BUF",
        projected_score=10.0,
    )
    genome = DraftStrategyGenome(
        projection_weight=0.0,
        position_scarcity_weight=0.0,
        adp_value_weight=1.0,
        upside_weight=0.0,
        floor_weight=0.0,
        bye_week_penalty=0.0,
        qb_priority=0.0,
        rb_priority=0.0,
        wr_priority=0.0,
        te_priority=0.0,
    )
    available_players = [weak_rb, strong_rb]

    agent = GenomeDraftAgent(genome=genome)

    strong_score = agent.score_player(strong_rb, available_players)
    weak_score = agent.score_player(weak_rb, available_players)

    assert strong_score > weak_score


def test_genome_draft_agent_can_complete_full_draft():
    players = create_fake_player_pool()
    teams = [Team(name=f"Team {number}") for number in range(1, 11)]
    league = League(
        name="Test League",
        teams=teams,
        available_players=players,
    )
    agent = GenomeDraftAgent(genome=create_test_genome())

    draft_results = run_snake_draft(
        league=league,
        rounds=16,
        draft_agent=agent,
    )

    assert len(draft_results) == 160

    for team in league.teams:
        assert team.roster_size() == 16
