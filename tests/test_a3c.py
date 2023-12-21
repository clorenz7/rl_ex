
import gymnasium as gym
import torch

from cor_rl import a3c

def test_n_step_returns():
    rewards = [2, 4, 8, 16, 32]
    gamma = 0.5

    final_value_est = 0
    n_step_returns = a3c.calc_n_step_returns(rewards, final_value_est, gamma)
    assert n_step_returns == [10, 16, 24, 32, 32]

    final_value_est = 64
    n_step_returns = a3c.calc_n_step_returns(rewards, final_value_est, gamma)
    assert n_step_returns == [12, 20, 32, 48, 64]


def test_cart_pole_eval():
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


def test_cart_pole_train():
    render_mode = "human" if False else 'rgb_array'
    env = gym.make('CartPole-v1', render_mode=render_mode)
    env.reset(seed=543)
    torch.manual_seed(543)

    agent_params = {
        'hidden_sizes': [128],
        # 'hidden_sizes': [24, 24],
        'n_actions': 2,
        'n_state': 4,
        'gamma': 0.99,
        'entropy_weight': 0.00,
        'grad_clip': 2.0,
    }
    train_params = {
        'optimizer': 'adamw',
        'lr': 3e-3,
        'weight_decay': 0.0,
    }
    global_agent = a3c.AdvantageActorCriticAgent(agent_params, train_params)
    torch.manual_seed(543)
    agents = [a3c.AdvantageActorCriticAgent(agent_params, train_params)]

    a3c.train_loop(
        global_agent, agents, [env],
        step_limit=1e9, episode_limit=2000, log_interval=10,
        solved_thresh=env.spec.reward_threshold
    )


def test_cart_pole_train_batched():
    render_mode = "human" if False else 'rgb_array'
    env = gym.make('CartPole-v1', render_mode=render_mode)
    env.reset(seed=543)
    torch.manual_seed(543)

    agent_params = {
        'hidden_sizes': [128],
        'n_actions': 2,
        'n_state': 4,
        'gamma': 0.99,
        'entropy_weight': 0.00,
        'grad_clip': 2.0,
    }
    train_params = {
        'optimizer': 'adamw',
        'lr': 1e-3,
        'weight_decay': 0.0,
    }
    global_agent = a3c.AdvantageActorCriticAgent(agent_params, train_params)
    torch.manual_seed(543)
    agents = [a3c.AdvantageActorCriticAgent(agent_params, train_params)]

    a3c.train_loop(
        global_agent, agents, [env],
        step_limit=1e9, episode_limit=2000, log_interval=10,
        solved_thresh=env.spec.reward_threshold, t_max=5
    )

def test_cart_pole_train_multi():
    torch.manual_seed(543)

    agent_params = {
        'hidden_sizes': [128],
        'n_actions': 2,
        'n_state': 4,
        'gamma': 0.99,
        'entropy_weight': 0.00,
        'grad_clip': 2.0,
    }
    train_params = {
        'optimizer': 'adamw',
        'lr': 1e-3,
        'weight_decay': 0.0,
    }
    global_agent = a3c.AdvantageActorCriticAgent(agent_params, train_params)

    agents, envs = [], []
    n_threads = 3

    for ii in range(n_threads):
        agents.append(
            a3c.AdvantageActorCriticAgent(agent_params, train_params),
        )
        envs.append(gym.make('CartPole-v1'))
        envs[-1].reset(seed=543 + 10 * ii)

    a3c.train_loop(
        global_agent, agents, envs,
        step_limit=1e9, episode_limit=2000, log_interval=10,
        solved_thresh=envs[0].spec.reward_threshold, t_max=5
    )


def test_cart_pole_train_arch():
    render_mode = "human" if False else 'rgb_array'
    env = gym.make('CartPole-v1', render_mode=render_mode)
    env.reset(seed=543)
    torch.manual_seed(543)

    agent_params = {
        # 'hidden_sizes': [128],
        'hidden_sizes': [32, 32],
        'n_actions': 2,
        'n_state': 4,
        'gamma': 0.99,
        'entropy_weight': 0.00,
        'grad_clip': 2.0,
    }
    train_params = {
        'optimizer': 'adamw',
        'lr': 1e-3 * 2,
        'weight_decay': 1e-4,
    }
    global_agent = a3c.AdvantageActorCriticAgent(agent_params, train_params)
    torch.manual_seed(543)
    agents = [a3c.AdvantageActorCriticAgent(agent_params, train_params)]

    a3c.train_loop(
        global_agent, agents, [env],
        step_limit=1e9, episode_limit=2000, log_interval=10,
        solved_thresh=env.spec.reward_threshold, t_max=1250
    )
