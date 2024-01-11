import os

import gymnasium as gym
import torch

import cor_rl.agents
from cor_rl import atari
from cor_rl import environments
from cor_rl.agents import atari_a2c
from cor_rl import a3c

RIGHT_FIRE = 4
NO_OP = 0

DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "rl_results")



def test_space_invaders_train():
    # TODO: Move this to a script

    agent_params = {
        'n_actions': 6,
        'gamma': 0.99,
        'entropy_weight': 0.01,  # 0.05 saw things happening...
        'grad_clip': 5.0,
        'type': 'a2c-atari',
        'reward_clip': 1.0
    }
    train_params = {
        'optimizer': 'rmsprop',
        'lr': 1e-3,
        'alpha': 0.99,
    }

    env_name = 'ALE/SpaceInvaders-v5'

    env_params = {
        'env_name': env_name,
        'reward_clip': 1.0,
        'repeat_action_probability': 0.0,
    }
    n_workers = 4

    torch.manual_seed(8888101)

    print("")
    a3c.train_loop_parallel(
        n_workers, agent_params, train_params, env_params,
        log_interval=10, seed=888, total_step_limit=500000,
        steps_per_batch=5, avg_decay=0.95,  # max_ep_steps=1e9,
        serial=False, render=True
    )
    import ipdb; ipdb.set_trace()


def test_space_invaders_a3c():
    agent_params = {
        'n_actions': 6,
        'gamma': 0.99,
        'entropy_weight': 0.01,
        'clip_grad_norm': 50.0,
        'type': 'a2c-atari',
        'reward_clip': 1.0,
        # 'n_exp_steps': 5.0,
    }
    train_params = {
        'optimizer': 'rmsprop',
        # 'lr': 8e-4,
        # 'lr': 1e-3,
        'lr': 1e-4 * 0.3,
        'alpha': 0.99,
    }

    env_name = 'ALE/SpaceInvaders-v5'

    env_params = {
        'env_name': env_name,
        'reward_clip': 1.0,
        'repeat_action_probability': 0.0,
    }
    n_workers = 8

    # print("")  # For ease of reading
    # agent, solved = a3c.train_loop_parallel(
    #     n_workers, agent_params, train_params, env_params,
    #     # log_interval=500, seed=8888101888,
    #     log_interval=100, seed=8888101888,
    #     # log_interval=10, seed=8888,
    #     total_step_limit=5e9,
    #     steps_per_batch=5, avg_decay=0.95,
    #     out_dir=os.path.join(DEFAULT_DIR, "a3c_test"),
    #     eval_interval=0.25, save_interval=2, use_mlflow=True,
    #     # use_mlflow=False, serial=True
    # )

    print("")  # For ease of reading
    env_params.pop('reward_clip')
    agent, solved = a3c.train_loop_continuous(
        n_workers, agent_params, train_params, env_params,
        # log_interval=500, seed=8888101888,
        log_interval=200, seed=8888101888,
        # log_interval=10, seed=8888,
        total_step_limit=5e9,
        steps_per_batch=5, avg_decay=0.95,
        out_dir=os.path.join(DEFAULT_DIR, "a3c_test"),
        eval_interval=0.25, save_interval=2, use_mlflow=True,
        # use_mlflow=False, serial=True
    )

    import ipdb; ipdb.set_trace()


def test_space_invaders_render():
    agent_params = {
        'n_actions': 6,
        'gamma': 0.99,
        'entropy_weight': 0.01,
        'clip_grad_norm': 50.0,
        'type': 'a2c-atari',
        'reward_clip': 1.0,
    }
    train_params = {
        'optimizer': 'adam',
        'lr': 0,
    }
    env_name = 'ALE/SpaceInvaders-v5'

    env_params = {
        'env_name': env_name,
        'repeat_action_probability': 0.0,
    }
    n_workers = 1
    # Cherry picked
    # seed = 1018888
    # load_file = os.path.join(
    #     DEFAULT_DIR, "a3c_test",
    #     '2024_Jan_10_H10_32_epoch12.pt'
    # )
    seed = 101  # Cherry picked. Gets a score of 920
    load_file = os.path.join(
        DEFAULT_DIR, "a3c_test",
        '2024_Jan_10_H10_32_epoch16.pt'
    )
    gif_file = os.path.join(
        DEFAULT_DIR, "a3c_test",
        '2024_Jan_10_H10_32_epoch16.gif'
    )

    print("")  # For ease of reading
    agent, solved = a3c.train_loop_continuous(
        n_workers, agent_params, train_params, env_params,
        log_interval=1, seed=seed,
        episode_limit=1,
        steps_per_batch=5000000, avg_decay=0.0,
        load_file=load_file,
        use_mlflow=False, serial=True, use_lock=False,
        save_gif=gif_file, render=False,
    )


def test_space_invaders_adam():
    agent_params = {
        'n_actions': 6,
        'gamma': 0.99,
        'entropy_weight': 0.01,
        'clip_grad_norm': 50.0,
        'type': 'a2c-atari',
        'reward_clip': 1.0,
    }
    train_params = {
        'optimizer': 'adam',
        'lr': 1e-4,
    }

    env_name = 'ALE/SpaceInvaders-v5'

    env_params = {
        'env_name': env_name,
        'repeat_action_probability': 0.0,
    }
    n_workers = 6

    print("")  # For ease of reading
    agent, solved = a3c.train_loop_continuous(
        n_workers, agent_params, train_params, env_params,
        log_interval=200, seed=8888101888,
        total_step_limit=50e6,
        steps_per_batch=5, avg_decay=0.95,
        out_dir=os.path.join(DEFAULT_DIR, "a3c_test"),
        eval_interval=0.25, save_interval=2, use_mlflow=True,
    )

    import ipdb; ipdb.set_trace()

