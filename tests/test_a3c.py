
import gymnasium as gym
import torch

from cor_rl.agents import a2c
from cor_rl import a3c
from cor_rl import utils


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
    agent = a2c.AdvantageActorCriticAgent(agent_params, train_params)

    result = a3c.agent_env_task(agent, agent, agent.optimizer, env, state=None)
    # Just make sure that we can evaluate the agent and get state
    assert 'state' in result


def test_cart_pole_train_pt_rep():
    """
    Test that cart pole can be solved similar to PyTorch reference
    implementation of A2C
    """

    agent_params = {
        'hidden_sizes': [128],
        'n_actions': 2,
        'n_state': 4,
        'gamma': 0.99,
        'entropy_weight': 0.00,
        'value_loss_clip': 1.0,
        'type': 'a2c-ffw',
        'norm_returns': True,
    }
    train_params = {
        'optimizer': 'adam',
        'lr': 3e-2,  # A learning rate of 3e-3 is more stable...
        'weight_decay': 0.0,
    }

    print("")
    agent, solved = a3c.train_loop_continuous(
        1, agent_params, train_params, 'CartPole-v1',
        total_step_limit=1e9, episode_limit=2000, log_interval=10,
        solved_thresh=gym.make('CartPole-v1').spec.reward_threshold,
        steps_per_batch=10000,
        debug=False, serial=True, seed=543,
        shared_mode=True, repro_mode=True
    )

    assert solved


def test_cart_pole_train_pt_rep_rmsprop():
    """
    Test that cart pole can be solved similar to PyTorch reference
    implementation of A2C
    """

    agent_params = {
        'hidden_sizes': [128],
        'n_actions': 2,
        'n_state': 4,
        'gamma': 0.99,
        'entropy_weight': 0.00,
        'value_loss_clip': 1.0,
        'type': 'a2c-ffw',
        'norm_returns': True,
    }
    train_params = {
        'optimizer': 'rmsprop',
        'lr': 1e-3,
        'weight_decay': 0.0,
    }

    n_failed = 0
    n_runs = 1
    for i in range(n_runs):
        print("")
        agent, solved = a3c.train_loop_continuous(
            4, agent_params, train_params, 'CartPole-v1',
            total_step_limit=1e9, episode_limit=2000, log_interval=10,
            solved_thresh=gym.make('CartPole-v1').spec.reward_threshold,
            steps_per_batch=10000,
            debug=False, serial=False, seed=543,
            shared_mode=True, repro_mode=True, use_lock=True
        )
        if not solved:
            n_failed += 1
    if n_failed > 0:
        print(f'# of failures: {n_failed}')

    assert solved


def test_cart_pole_train_batched():
    """
    Test that cart pole can be solved with small batches of experience
    """
    render_mode = "human" if False else 'rgb_array'
    env = gym.make('CartPole-v1', render_mode=render_mode)
    seed = 543
    env.reset(seed=seed)
    torch.manual_seed(seed)

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
        'optimizer': 'adam',
        'lr': 1e-3,
        'weight_decay': 0.0,
    }
    steps_per_batch = 125

    print("")
    agent, solved = a3c.train_loop_continuous(
        1, agent_params, train_params, 'CartPole-v1',
        total_step_limit=1e9, episode_limit=2000, log_interval=100,
        solved_thresh=gym.make('CartPole-v1').spec.reward_threshold,
        steps_per_batch=steps_per_batch, seed=seed,
        debug=False, serial=True,
        shared_mode=True
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
        'optimizer': 'adam',
        'lr': 1e-3 * 2,
    }
    print("")
    agent, solved = a3c.train_loop_continuous(
        1, agent_params, train_params, 'CartPole-v1',
        total_step_limit=1e9, episode_limit=2000, log_interval=100,
        solved_thresh=gym.make('CartPole-v1').spec.reward_threshold,
        steps_per_batch=1250,
        debug=False, serial=True, seed=543, shared_mode=True
    )

    assert solved


def test_cart_pole_train_multi():
    """
    Test that multiple agents on a single thread can solve cart pole
    """
    seed = 543101234
    torch.manual_seed(seed)

    agent_params = {
        'type': 'a2c-ffw',
        'hidden_sizes': [128],
        'n_actions': 2,
        'n_state': 4,
        'gamma': 0.99,
        'entropy_weight': 0.00,
        'value_loss_clip': 1.0,
    }
    train_params = {
        'optimizer': 'adam',
        'lr': 1e-3,
        'weight_decay': 0.0,
    }

    n_threads = 16
    env_name = 'CartPole-v1'

    steps_per_batch = 15 * 10

    print("")
    agent, solved = a3c.train_loop_continuous(
        n_threads, agent_params, train_params, env_name,
        total_step_limit=1e9, episode_limit=2000, log_interval=10,
        solved_thresh=gym.make(env_name).spec.reward_threshold,
        steps_per_batch=steps_per_batch,
        debug=False, seed=seed,
        serial=True, shared_mode=True
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
        'optimizer': 'adam',
        'lr': 8e-4,
        'weight_decay': 0.0,
    }

    env_name = 'CartPole-v1'
    n_workers = 4

    print("")  # For ease of reading
    agent, solved = a3c.train_loop_continuous(
        n_workers, agent_params, train_params, env_name,
        log_interval=100, seed=543, total_step_limit=1e9, episode_limit=2000,
        solved_thresh=450, steps_per_batch=10000, avg_decay=0.95,
        debug=False, use_mlflow=False, use_lock=True
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
