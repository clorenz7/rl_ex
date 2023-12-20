from cor_rl import a3c


def test_n_step_returns():
    rewards = [2, 4, 8, 16, 32]
    gamma = 0.5

    n_step_returns = a3c.calc_n_step_returns(rewards, gamma)

    assert n_step_returns == [10, 16, 24, 32, 32]
