from evolution.full_season import calculate_full_season_fitness


def test_full_season_fitness_rewards_wins_playoffs_championship_and_transactions():
    score = calculate_full_season_fitness(
        regular_season_wins=10,
        points_for=1000.0,
        playoff_seed=1,
        playoff_wins=2,
        champion=True,
        transaction_reward=40.0,
    )

    assert score == 375.0


def test_full_season_fitness_punishes_negative_transaction_value():
    positive = calculate_full_season_fitness(5, 800.0, None, 0, False, 20.0)
    negative = calculate_full_season_fitness(5, 800.0, None, 0, False, -20.0)

    assert positive > negative
