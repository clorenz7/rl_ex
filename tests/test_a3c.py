from cor_rl import a3c
import gymnasium as gym

def test_n_step_returns():
    rewards = [2, 4, 8, 16, 32]
    gamma = 0.5

    final_value_est = 0
    n_step_returns = a3c.calc_n_step_returns(rewards, final_value_est, gamma)
    assert n_step_returns == [10, 16, 24, 32, 32]

    final_value_est = 64
    n_step_returns = a3c.calc_n_step_returns(rewards, final_value_est, gamma)
    assert n_step_returns == [12, 20, 32, 48, 64]


def test_cart_pole():
    render_mode = "human" if True else 'rgb_array'
    env = gym.make('CartPole-v1', render_mode=render_mode)
    agent_params = {
        'hidden_sizes': [128],
        'n_actions': 2,
        'n_state': 4
    }
    train_params = {
        'optimizer': 'rmsprop'
    }
    agent = a3c.AdvantageActorCriticAgent(agent_params, train_params)

    result = a3c.agent_env_task(agent, env, parameters=None, state=None)

    import ipdb; ipdb.set_trace()