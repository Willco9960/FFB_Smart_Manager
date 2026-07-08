from main import APP_NAME, APP_VERSION, main


def test_main_runs(capsys):
    main()

    captured = capsys.readouterr()

    assert APP_NAME in captured.out
    assert APP_VERSION in captured.out
    assert "ready for local development" in captured.out
    assert "Fake player pool loaded: 188 players" in captured.out
    assert "Draft completed: 160 picks" in captured.out
    assert "Draft agent: RandomDraftAgent" in captured.out
