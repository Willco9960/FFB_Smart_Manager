from main import APP_NAME, APP_VERSION, main


def test_main_runs(capsys):
    main()

    captured = capsys.readouterr()

    assert APP_NAME in captured.out
    assert APP_VERSION in captured.out
    assert "ready for local development" in captured.out
