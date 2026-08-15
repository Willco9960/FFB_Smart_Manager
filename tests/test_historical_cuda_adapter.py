import torch

from gpu_sim.historical_adapter import create_historical_cuda_inputs


def test_historical_cuda_adapter_uses_previous_season_projection_and_weekly_shapes(
    tmp_path,
    monkeypatch,
):
    class _Player:
        def __init__(self, name, position, team, projected_score, actual_score):
            self.name = name
            self.position = position
            self.team = team
            self.projected_score = projected_score
            self.actual_score = actual_score

    class _Performance:
        def __init__(self, player_name, position, week, fantasy_points):
            self.player_name = player_name
            self.position = position
            self.week = week
            self.fantasy_points = fantasy_points

    players = [
        _Player(f"P{index}", position, "TST", 100.0 - index, 10.0)
        for index in range(160)
        for position in ("QB", "RB", "WR", "TE")
    ][:160]
    performances = [
        _Performance("P0", "QB", 1, 25.0),
        _Performance("P0", "QB", 2, 30.0),
    ]
    monkeypatch.setattr(
        "gpu_sim.historical_adapter.load_player_stats",
        lambda season, raw_data_dir: [],
    )
    monkeypatch.setattr(
        "gpu_sim.historical_adapter.load_leakage_safe_player_pool",
        lambda **kwargs: players,
    )
    monkeypatch.setattr(
        "gpu_sim.historical_adapter.load_weekly_performances",
        lambda **kwargs: performances,
    )

    inputs = create_historical_cuda_inputs(
        season=2021,
        players=160,
        weeks=17,
        raw_data_dir=tmp_path,
    )

    assert inputs.projection_season == 2020
    assert inputs.state.draft_projections.shape == (1, 160)
    assert inputs.state.weekly_projections.shape == (1, 17, 160)
    assert inputs.state.weekly_actual_points.shape == (1, 17, 160)
    assert torch.isclose(inputs.state.weekly_actual_points[0, 0, 0], torch.tensor(25.0))
