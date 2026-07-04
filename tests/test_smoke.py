from main import main


def test_main_runs(capsys):
    main()

    captured = capsys.readouterr()

    assert "Fantasy Football AI Manager setup is working." in captured.out
