from scripts.run_cuda_cpu_historical_comparison import parse_args


def test_comparison_script_has_expected_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", ["comparison"])
    args = parse_args()

    assert args.start_season == 2021
    assert args.end_season == 2024
    assert args.transactions is False
