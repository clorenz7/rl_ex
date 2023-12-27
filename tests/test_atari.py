import gymnasium as gym
import torch

from cor_rl import atari
from cor_rl import environments
from cor_rl.agents import atari_a2c

RIGHT_FIRE = 4
NO_OP = 0


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
    assert len(env.frame_buffer) == 1
    # There should be no frames returned on termination
    assert len(frames) == 0
