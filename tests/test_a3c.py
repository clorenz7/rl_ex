
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

    old = False

    if old:
        print("Serial Code!")
        global_agent = cor_rl.agents.factory(agent_params, train_params)
        torch.manual_seed(543)
        agents = [cor_rl.agents.factory(agent_params, train_params)]

        agent, solved = a3c.train_loop(
            global_agent, agents, [env],
            total_step_limit=1e9, episode_limit=2000, log_interval=100,
            solved_thresh=env.spec.reward_threshold, seed=543
        )

    else:
        print("")
        print("Parallel Code!")
        agent, solved = a3c.train_loop_parallel(
            1, agent_params, train_params, 'CartPole-v1',
            total_step_limit=1e9, episode_limit=2000, log_interval=100,
            solved_thresh=gym.make('CartPole-v1').spec.reward_threshold,
            steps_per_batch=10000000,
            debug=False, serial=True, seed=543, shared_mode=False
        )

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
        'optimizer': 'adamw',
        'lr': 1e-3,
        'weight_decay': 0.0,
    }

    global_agent = cor_rl.agents.factory(agent_params, train_params)
    torch.manual_seed(seed)
    agents = [cor_rl.agents.factory(agent_params, train_params)]

    old = False
    steps_per_batch = 125
    if old:
        print("Serial Code!")
        agent, solved = a3c.train_loop(
            global_agent, agents, [env],
            total_step_limit=1e9, episode_limit=2000, log_interval=100,
            solved_thresh=env.spec.reward_threshold,
            steps_per_batch=steps_per_batch, seed=seed
        )
    else:
        print("")
        print("Parallel Code!")
        agent, solved = a3c.train_loop_parallel(
            1, agent_params, train_params, 'CartPole-v1',
            total_step_limit=1e9, episode_limit=2000, log_interval=100,
            solved_thresh=gym.make('CartPole-v1').spec.reward_threshold,
            steps_per_batch=steps_per_batch, seed=seed,
            debug=False, serial=True,  shared_mode=False
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

    old = False
    if old:
        agent, solved = a3c.train_loop(
            global_agent, agents, [env],
            total_step_limit=1e9, episode_limit=2000, log_interval=100,
            solved_thresh=env.spec.reward_threshold, steps_per_batch=1250,
            seed=543
        )
    else:
        print("")
        print("Parallel Code!")
        agent, solved = a3c.train_loop_parallel(
            1, agent_params, train_params, 'CartPole-v1',
            total_step_limit=1e9, episode_limit=2000, log_interval=100,
            solved_thresh=gym.make('CartPole-v1').spec.reward_threshold,
            steps_per_batch=1250,
            debug=False, serial=True, seed=543, shared_mode=False
        )

    assert solved


def test_cart_pole_train_multi():
    """
    Test that multiple agents on a single thread can solve cart pole
    """
    torch.manual_seed(543)

    agent_params = {
        'type': 'a2c-ffw',
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

    agents, envs = [], []
    n_threads = 3
    env_name = 'CartPole-v1'

    old = False
    steps_per_batch = 125

    if old:
        global_agent = a3c.AdvantageActorCriticAgent(agent_params, train_params)
        for ii in range(n_threads):
            t_seed = 543 + 10 * ii
            # t_seed = 543 + 101 * ii
            torch.manual_seed(t_seed)
            agents.append(
                a3c.AdvantageActorCriticAgent(agent_params, train_params),
            )
            envs.append(gym.make(env_name))
            envs[-1].reset(seed=t_seed)

        # ipdb> agents[1].net.base_layer[0].weight[:2,:]
        # tensor([[ 0.0604, -0.3619,  0.2966, -0.4935],
        #         [ 0.2108,  0.0493,  0.3187, -0.1849]], grad_fn=<SliceBackward0>)
        agent, solved = a3c.train_loop(
            global_agent, agents, envs,
            total_step_limit=1e9, episode_limit=2000, log_interval=100,
            solved_thresh=envs[0].spec.reward_threshold,
            steps_per_batch=steps_per_batch, seed=543,
            debug=False
        )
        # After 1 batch:
        # ipdb> states[0]
        # array([ 0.01445378,  0.18729304, -0.03212173, -0.37609693], dtype=float32)
        # ipdb> states[1]
        # array([ 0.00486106,  0.971259  , -0.04166358, -1.4610286 ], dtype=float32)
        # Progression:
        # tests/test_a3c.py::test_cart_pole_train_multi
        # Episode 100   Max reward: 64.00   Average reward: 14.95    Time: 0.0min
        # Episode 200   Max reward: 36.00   Average reward: 11.92    Time: 0.0min
        # Episode 300   Max reward: 120.00  Average reward: 20.41    Time: 0.0min
        # Episode 400   Max reward: 331.00  Average reward: 88.80    Time: 0.1min
        # Episode 500   Max reward: 440.00  Average reward: 75.87    Time: 0.1min
        # Episode 600   Max reward: 1141.00 Average reward: 203.61   Time: 0.2min
        # Episode 700   Max reward: 2962.00 Average reward: 472.54   Time: 0.4min
        # Episode 770   Last reward: 5013.00    Average reward: 478.42
        # PROBLEM SOLVED in 29.3sec
    else:
        print("")
        agent, solved = a3c.train_loop_parallel(
            n_threads, agent_params, train_params, env_name,
            total_step_limit=1e9, episode_limit=2000, log_interval=100,
            solved_thresh=gym.make(env_name).spec.reward_threshold,
            steps_per_batch=steps_per_batch,
            debug=False, serial=True, seed=543, shared_mode=False
        )
        # After 1 batch:
        # ipdb> msg_pipes[0].worker.state
        # array([-0.00115787, -0.20325479, -0.00865413,  0.2161426 ], dtype=float32)
        # ipdb> msg_pipes[1].worker.state
        # array([-0.04978548, -0.5908184 ,  0.04057004,  0.90578496], dtype=float32)

        # Initial Selections:
        # (tensor(-0.6909, grad_fn=<SqueezeBackward1>), 0)
        # (tensor(-0.7129, grad_fn=<SqueezeBackward1>), 1)
        # (tensor(-0.6909, grad_fn=<SqueezeBackward1>), 0)
        # (tensor(-0.6746, grad_fn=<SqueezeBackward1>), 0)
        # (tensor(-0.7389, grad_fn=<SqueezeBackward1>), 1)
        # tensor([-0.1966, -0.4916,  0.2373,  0.1841], grad_fn=<SliceBackward0>)
        # (tensor(-0.6887, grad_fn=<SqueezeBackward1>), 0)
        # (tensor(-0.7181, grad_fn=<SqueezeBackward1>), 1)
        # (tensor(-0.6884, grad_fn=<SqueezeBackward1>), 0)
        # (tensor(-0.6676, grad_fn=<SqueezeBackward1>), 0)
        # (tensor(-0.6399, grad_fn=<SqueezeBackward1>), 0)
        # tensor([-0.1970, -0.4926,  0.2372,  0.1850], grad_fn=<SliceBackward0>)
        # (tensor(-0.7008, grad_fn=<SqueezeBackward1>), 1)
        # (tensor(-0.6945, grad_fn=<SqueezeBackward1>), 1)
        # (tensor(-0.6963, grad_fn=<SqueezeBackward1>), 1)
        # (tensor(-0.6854, grad_fn=<SqueezeBackward1>), 0)
        # (tensor(-0.6942, grad_fn=<SqueezeBackward1>), 1)
        # tensor([-0.1973, -0.4932,  0.2370,  0.1856], grad_fn=<SliceBackward0>)
        # (tensor(-0.6378, grad_fn=<SqueezeBackward1>), 0)
        # (tensor(-0.7783, grad_fn=<SqueezeBackward1>), 1)
        # (tensor(-0.7522, grad_fn=<SqueezeBackward1>), 1)
        # (tensor(-0.6507, grad_fn=<SqueezeBackward1>), 0)
        # (tensor(-0.6375, grad_fn=<SqueezeBackward1>), 0)
        # tensor([-0.1975, -0.4940,  0.2368,  0.1862], grad_fn=<SliceBackward0>)

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
        debug=False, use_mlflow=False
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


# def test_cart_pole_train_a3c_accum():
#     """
#     Test that the multi-threaded version of the code can solve cart pole
#     """
#     agent_params = {
#         'hidden_sizes': [128],
#         'n_actions': 2,
#         'n_state': 4,
#         'gamma': 0.99,
#         'entropy_weight': 0.00,
#         'value_loss_clip': 2.0,
#         'type': 'a2c-ffw',
#     }
#     train_params = {
#         'optimizer': 'adamw',
#         'lr': 4e-3,
#         'weight_decay': 0.0,
#     }

#     env_name = 'CartPole-v1'
#     n_workers = 4

#     print("")  # For ease of reading
#     agent, solved = a3c.train_loop_parallel(
#         n_workers, agent_params, train_params, env_name,
#         log_interval=100, seed=543, total_step_limit=1e9, episode_limit=2000,
#         solved_thresh=450, steps_per_batch=10000, avg_decay=0.95,
#         debug=False, accumulate_grads=True
#     )
#     assert solved

#     make_gif = False
#     if make_gif:
#         import os
#         render_mode = "human" if False else 'rgb_array'
#         env = gym.make('CartPole-v1', render_mode=render_mode)
#         env.reset(seed=101)
#         result = a3c.agent_env_task(
#             agent, env, parameters=None, state=None, t_max=1000,
#             output_frames=True
#         )

#         file_name = os.path.join(utils.DEFAULT_DIR, "a3c_cart_pole.gif")
#         utils.write_gif(result['frames'][::5], file_name, duration=1)
