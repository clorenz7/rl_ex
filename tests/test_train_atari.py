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
    }
    train_params = {
        'optimizer': 'rmsprop',
        'lr': 1e-3,
        'alpha': 0.99,
    }

    env_name = 'ALE/SpaceInvaders-v5'
    n_workers = 4

    torch.manual_seed(8888101)

    global_agent = cor_rl.agents.factory(agent_params, train_params)
    agents = []
    envs = []
    for i in range(n_workers):
        env = environments.factory(env_name, render_mode='human' if i == 0 else None)
        env.reset(seed=888 + i * 101)
        envs.append(env)
        agents.append(cor_rl.agents.factory(agent_params, train_params))

    agent, solved = a3c.train_loop(
        global_agent, agents, envs,
        log_interval=10, seed=888, total_step_limit=500000,
        steps_per_batch=5, avg_decay=0.95, max_ep_steps=1e9
    )
    # print("")  # For ease of reading
    # agent, solved = a3c.train_loop(
    #     n_workers, agent_params, train_params, env_name,
    #     log_interval=10, seed=888, total_step_limit=50000,
    #     steps_per_batch=5, avg_decay=0.95,
    # )
    import ipdb; ipdb.set_trace()


def test_space_invaders_a3c():
    agent_params = {
        'n_actions': 6,
        'gamma': 0.99,
        'entropy_weight': 0.01,
        'clip_grad_norm': 0.02,
        'type': 'a2c-atari',
        'reward_clip': 1.0,
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
    n_workers = 8

    print("")  # For ease of reading
    agent, solved = a3c.train_loop_parallel(
        n_workers, agent_params, train_params, env_params,
        log_interval=500, seed=8888101888,
        total_step_limit=5e9,
        steps_per_batch=5, avg_decay=0.95,
        eval_interval=0.05, save_interval=0.05,
        out_dir=os.path.join(DEFAULT_DIR, "a3c_test"),
        use_mlflow=False
    )
    import ipdb; ipdb.set_trace()