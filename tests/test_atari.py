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


def test_frame_preprocessing():

    env = gym.make("ALE/SpaceInvaders-v5")
    prev_frame, _ = env.reset(seed=101)
    frame, reward, terminated, trunc, info = env.step(0)

    comp_frame = atari.preprocess_frames(frame, prev_frame)
    # Test that frame is correct size
    assert comp_frame.shape == (84, 84)
    # Test that frame has correct scale
    assert comp_frame.max() <= 1
    # Test that frame has content
    assert comp_frame.std() > 27.0/255


def test_atari_env_wrapper():
    """
    Test wrapper around Atari env which stacks and preprocesses frames
    """
    env = environments.factory.get("ALE/SpaceInvaders-v5")
    # Check that wrapper gets the action space and spec properly
    assert env.action_space.n == 6
    assert env.spec.reward_threshold is None

    init_frame, info = env.reset(seed=10101)

    # Check that initial frame is correct  shape and repeated
    assert init_frame.shape == (4, 84, 84)
    assert torch.allclose(
        init_frame[0, :, :], init_frame[3, :, :],
        rtol=1e-4, atol=1e-4
    )
    # Check that preprocessing is correct
    assert init_frame.min() >= 0
    assert init_frame.max() <= 1
    assert init_frame.std() > 27.0/255

    # Check that we can perform an action
    frames, reward, terminated, trunc, info = env.step(RIGHT_FIRE)
    assert frames.shape == (4, 84, 84)
    assert reward == 0
    assert terminated is False

    # Just move right and fire for a few steps, you should randomly
    # hit something and the frames should change over time
    total_reward = 0
    for _ in range(60):
        total_reward += reward
        frames, reward, terminated, trunc, info = env.step(RIGHT_FIRE)

    frame_diff = frames[3, :, :] - frames[0, :, :]

    assert total_reward > 0
    assert frame_diff.std() > 0.01
    assert terminated is False


def test_atari_net():
    """
    Test that the network can process the stacked frames
    """
    env = environments.factory.get("ALE/SpaceInvaders-v5")
    init_frame, info = env.reset(seed=10101)
    torch.manual_seed(10101)

    net = atari_a2c.Mnih2015PolicyValueNetwork(env.action_space.n)

    action_probs, value_est = net.forward(init_frame)

    assert action_probs.numel() == env.action_space.n
    assert abs(action_probs.sum().item() - 1.0) < 1e-6
    assert value_est.numel() == 1


def test_life_counter_ends_episode():
    env = environments.factory.get("ALE/SpaceInvaders-v5")
    init_frame, info = env.reset(seed=10101)
    n_start_lives = info.get('lives')
    stayin_alive = True
    n_steps = 0
    max_steps = 1000

    all_frames = []
    while stayin_alive and n_steps < max_steps:
        frames, reward, terminated, trunc, info = env.step(RIGHT_FIRE)
        stayin_alive = info.get('lives') == n_start_lives
        n_steps += 1
        if stayin_alive:
            all_frames.append(frames)

    assert stayin_alive is False
    assert terminated is False
    assert info.get('lives') < n_start_lives
    assert n_steps < 350
    # At episode start, there should be one frame in the buffer
    assert len(env.frame_buffer) > 0
    # There should be no frames returned on termination
    assert frames is None


# def test_compare_envs():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     env_name = 'ALE/SpaceInvaders-v5'
#     cor_env = environments.factory(env_name, noop_max=1)
#     c_state, _ = cor_env.reset(seed=8888)

#     wrap_env = environments.EnvironmentFactory(
#         use_gym_atari_wrapper=True
#     ).get(env_name, noop_max=1)
#     w_state , _ = wrap_env.reset(seed=8888)

#     c_states, w_states = [], []
#     w_buf = [w_state] * 4
#     for i in range(60):
#         c_state, _, _, _, _ = cor_env.step(RIGHT_FIRE)
#         c_states.append(c_state.numpy())

#         for ii in range(4):
#             w_state, _, _, _, _ = wrap_env.step(RIGHT_FIRE)
#             w_buf.append(w_state)
#             w_buf = w_buf[-4:]
#             w_states.append(np.transpose(np.dstack(w_buf), [2,0,1]))

#     idx = 45;
#     plt.subplot(2, 1,1); plt.imshow(np.hstack([c_states[idx][i, :, :] for i in range(4)]));
#     plt.subplot(2, 1,2); plt.imshow(np.hstack([w_states[idx][i, :, :] for i in range(4)]));
#     plt.show()
#     import ipdb; ipdb.set_trace()


def _compare_frames(stack_1, stack_2):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.subplot(2, 1, 1);
    plt.imshow(np.hstack([stack_1[i, :, :] for i in range(4)]));
    plt.subplot(2, 1, 2);
    plt.imshow(np.hstack([stack_2[i, :, :] for i in range(4)]));
    plt.show()


def test_compare_envs():

    env_name = 'ALE/SpaceInvaders-v5'
    cor_env = environments.factory(
        env_name, noop_max=1, reward_clip=None, repeat_action_probability=0.0
    )
    c_state, _ = cor_env.reset(seed=8888)

    wrap_env = environments.factory(
        'W'+env_name, noop_max=1, repeat_action_probability=0.0,
    )
    w_state, _ = wrap_env.reset(seed=8888)
    # Extra no nops to match cor_env inital no_op step
    for _ in range(4):
        wrap_env.env.env.step(NO_OP)

    c_states, w_states = [], []
    c_rewards, w_rewards = [], []

    terminated = False

    while not terminated:
        c_state, reward, c_term, _, _ = cor_env.step(RIGHT_FIRE)
        c_rewards.append(reward)
        if c_state is not None:
            c_states.append(c_state.numpy())

        w_state, reward, w_term, _, _ = wrap_env.step(RIGHT_FIRE)
        w_rewards.append(reward)
        if w_state is not None:
            w_states.append(w_state.numpy())

        terminated = c_term or w_term

    assert c_rewards == w_rewards
    assert c_term and w_term
    assert abs(c_states[50].mean() - w_states[50].mean()) < 1e-3
    assert abs(c_states[50].std() - w_states[50].std()) < 1e-3

