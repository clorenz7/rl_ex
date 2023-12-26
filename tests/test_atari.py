import gymnasium as gym
import torch

from cor_rl import factories
from cor_rl import atari

RIGHT_FIRE = 4


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


def test_space_invaders():
    env = factories.environment_factory.get("ALE/SpaceInvaders-v5")
    # Check that wrapper gets the action space and spec properly
    assert env.action_space.n == 6
    assert env.spec.reward_threshold is None

    init_frame, info = env.reset(seed=10101)

    # Check that initial frame is correct  shape and repeated
    assert init_frame.shape == (84, 84, 4)
    assert torch.allclose(
        init_frame[:, :, 0], init_frame[:, :, 3],
        rtol=1e-4, atol=1e-4
    )
    # Check that preprocessing is correct
    assert init_frame.min() >= 0
    assert init_frame.max() <= 1
    assert init_frame.std() > 27.0/255

    # Check that we can perform an action
    frames, reward, terminated, trunc, info = env.step(RIGHT_FIRE)
    assert frames.shape == (84, 84, 4)
    assert reward == 0
    assert terminated is False

    # Just move right and fire for a few steps, you should randomly
    # hit something and the frames should change over time
    total_reward = 0
    for _ in range(15):
        total_reward += reward
        frames, reward, terminated, trunc, info = env.step(RIGHT_FIRE)

    frame_diff = frames[:, :, 3] - frames[:, :, 0]

    assert total_reward > 0
    assert frame_diff.std() > 0.01
    assert terminated is False

