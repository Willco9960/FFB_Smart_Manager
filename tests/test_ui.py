from ui.app import VIEW_BY_KEY, VIEW_DEFINITIONS
from ui.demo_content import LINEUP_PLAYER_ANALYTICS, TOP_FREE_AGENT_PROSPECTS, TOP_TRADE_PROSPECTS
from ui.league_state import DEMO_LEAGUES, LeagueWorkspace


def test_ui_defines_all_major_views():
    expected_views = {"home", "draft", "lineup", "waivers", "trades", "models", "settings"}

    assert set(VIEW_BY_KEY) == expected_views
    assert {view.key for view in VIEW_DEFINITIONS} == expected_views


def test_ui_view_definitions_have_titles_and_descriptions():
    for view in VIEW_DEFINITIONS:
        assert view.title
        assert view.subtitle
        assert view.description


def test_demo_workspace_supports_multiple_platforms():
    workspace = LeagueWorkspace()

    assert len(workspace.leagues) == 4
    assert workspace.platforms == ("ESPN", "Sleeper", "NFL Fantasy")
    assert {league.platform for league in DEMO_LEAGUES} == set(workspace.platforms)


def test_workspace_can_focus_and_clear_a_league():
    workspace = LeagueWorkspace()

    selected = workspace.select("sleeper-dynasty")

    assert selected.platform == "Sleeper"
    assert workspace.active_league == selected

    workspace.clear_selection()

    assert workspace.active_league is None


def test_lineup_demo_has_unique_starters_and_alternatives():
    starter_names = [player.name for player in LINEUP_PLAYER_ANALYTICS]

    assert len(starter_names) == len(set(starter_names))
    assert LINEUP_PLAYER_ANALYTICS[0].position == "QB"
    assert TOP_FREE_AGENT_PROSPECTS
    assert TOP_TRADE_PROSPECTS
