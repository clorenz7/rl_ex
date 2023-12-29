
import gymnasium as gym
import torch

from cor_rl.agents import a2c
from cor_rl import a3c
from cor_rl import utils
import cor_rl.agents


def test_n_step_returns():
    """
    Test that n step returns are calcualted as expected
    """
    rewards = [2, 4, 8, 16, 32]
    gamma = 0.5

    final_value_est = 0
    n_step_returns = a2c.calc_n_step_returns(rewards, final_value_est, gamma)
    assert n_step_returns == [10, 16, 24, 32, 32]

    final_value_est = 64
    n_step_returns = a2c.calc_n_step_returns(rewards, final_value_est, gamma)
    assert n_step_returns == [12, 20, 32, 48, 64]


def test_cart_pole_eval():
    """
    Test that you can get a single update from an agent
    """
    render_mode = "human" if False else 'rgb_array'
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
    # Just make sure that we can evaluate the agent and get grads
    assert 'grads' in result


def test_cart_pole_train_pt_rep():
    """
    Test that cart pole can be solved similar to PyTorch reference
    implementation of A2C
    """
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
        'value_loss_clip': 2.0,
        'type': 'a2c-ffw',
    }
    train_params = {
        'optimizer': 'adamw',
        'lr': 3e-3,
        'weight_decay': 0.0,
    }
    global_agent = cor_rl.agents.factory(agent_params, train_params)
    torch.manual_seed(543)
    agents = [cor_rl.agents.factory(agent_params, train_params)]

    agent, solved = a3c.train_loop(
        global_agent, agents, [env],
        total_step_limit=1e9, episode_limit=2000, log_interval=100,
        solved_thresh=env.spec.reward_threshold
    )
    assert solved


def test_cart_pole_train_batched():
    """
    Test that cart pole can be solved with small batches of experience
    """
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
        'value_loss_clip': 2.0,
        'type': 'a2c-ffw',
    }
    train_params = {
        'optimizer': 'adamw',
        'lr': 1e-3,
        'weight_decay': 0.0,
    }
    global_agent = cor_rl.agents.factory(agent_params, train_params)
    torch.manual_seed(543)
    agents = [cor_rl.agents.factory(agent_params, train_params)]

    agent, solved = a3c.train_loop(
        global_agent, agents, [env],
        total_step_limit=1e9, episode_limit=2000, log_interval=100,
        solved_thresh=env.spec.reward_threshold, steps_per_batch=5
    )
    assert solved


def test_cart_pole_train_arch():
    """
    Test that cart pole can be solved with an alternate architecture
    """
    render_mode = "human" if False else 'rgb_array'
    env = gym.make('CartPole-v1', render_mode=render_mode)
    env.reset(seed=543)
    torch.manual_seed(543)

    agent_params = {
        'hidden_sizes': [32, 32],
        'n_actions': 2,
        'n_state': 4,
        'gamma': 0.99,
        'entropy_weight': 0.00,
        'value_loss_clip': 2.0,
        'type': 'a2c-ffw',
    }
    train_params = {
        'optimizer': 'adamw',
        'lr': 1e-3 * 2,
        'weight_decay': 1e-4,
    }
    global_agent = cor_rl.agents.factory(agent_params, train_params)
    torch.manual_seed(543)
    agents = [cor_rl.agents.factory(agent_params, train_params)]

    agent, solved = a3c.train_loop(
        global_agent, agents, [env],
        total_step_limit=1e9, episode_limit=2000, log_interval=100,
        solved_thresh=env.spec.reward_threshold, steps_per_batch=1250
    )
    assert solved


def test_cart_pole_train_multi():
    """
    Test that multiple agents on a single thread can solve cart pole
    """
    torch.manual_seed(543)

    agent_params = {
        'hidden_sizes': [128],
        'n_actions': 2,
        'n_state': 4,
        'gamma': 0.99,
        'entropy_weight': 0.00,
        'value_loss_clip': 2.0,
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
        t_seed = 543 + 10 * ii
        torch.manual_seed(t_seed)
        agents.append(
            a3c.AdvantageActorCriticAgent(agent_params, train_params),
        )
        envs.append(gym.make('CartPole-v1'))
        envs[-1].reset(seed=t_seed)

    agent, solved = a3c.train_loop(
        global_agent, agents, envs,
        total_step_limit=1e9, episode_limit=2000, log_interval=100,
        solved_thresh=envs[0].spec.reward_threshold, steps_per_batch=5,
        debug=False
    )
    assert solved


def test_cart_pole_train_a3c():
    """
    Test that the multi-threaded version of the code can solve cart pole
    """
    agent_params = {
        'hidden_sizes': [128],
        'n_actions': 2,
        'n_state': 4,
        'gamma': 0.99,
        'entropy_weight': 0.00,
        'value_loss_clip': 2.0,
        'type': 'a2c-ffw',
    }
    train_params = {
        'optimizer': 'adamw',
        'lr': 4e-3,
        'weight_decay': 0.0,
    }

    env_name = 'CartPole-v1'
    n_workers = 4

    print("")  # For ease of reading
    agent, solved = a3c.train_loop_parallel(
        n_workers, agent_params, train_params, env_name,
        log_interval=100, seed=543, total_step_limit=1e9, episode_limit=2000,
        solved_thresh=450, steps_per_batch=10000, avg_decay=0.95,
        debug=False
    )
    assert solved

    make_gif = False
    if make_gif:
        import os
        render_mode = "human" if False else 'rgb_array'
        env = gym.make('CartPole-v1', render_mode=render_mode)
        env.reset(seed=101)
        result = a3c.agent_env_task(
            agent, env, parameters=None, state=None, t_max=1000,
            output_frames=True
        )

        file_name = os.path.join(utils.DEFAULT_DIR, "a3c_cart_pole.gif")
        utils.write_gif(result['frames'][::5], file_name, duration=1)
