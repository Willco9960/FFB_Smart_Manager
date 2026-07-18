from ui.app import VIEW_BY_KEY, VIEW_DEFINITIONS


def test_ui_defines_all_major_views():
    expected_views = {"home", "draft", "lineup", "waivers", "trades", "models", "settings"}

    assert set(VIEW_BY_KEY) == expected_views
    assert {view.key for view in VIEW_DEFINITIONS} == expected_views


def test_ui_view_definitions_have_titles_and_descriptions():
    for view in VIEW_DEFINITIONS:
        assert view.title
        assert view.subtitle
        assert view.description
