
import gymnasium as gym
import numpy as np
import skimage
import torch
from torch import nn

LUM_COEFFS = np.array([0.299, 0.587, 0.114])


class PolicyValueImageNetwork(nn.Module):
    def __init__(self, n_actions, n_channels=4):
        super().__init__()
        # Image size: 84 x 84 x n_channels
        n_filters_1 = 16
        filter_size_1 = (8, 8)
        stride_1 = 4

        n_filters_2 = 32
        filter_size_2 = (4, 4)
        stride_2 = 2

        n_features = 9 * 9 * n_filters_2
        n_hidden = 256

        self.base_net = nn.Sequential(
            nn.Conv2d(n_channels, n_filters_1, filter_size_1, stride_1),
            nn.ReLU(),
            nn.Conv2d(n_filters_1, n_filters_2, filter_size_2, stride_2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(n_hidden, n_actions),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.base_net(x)
        action_probs = self.policy_head(x)
        value_est = self.value_head(x)

        return action_probs, value_est


def preprocess_frames(frame, prev_frame, out_size=(84, 84)):
    composite_frame = np.zeros(frame.shape)
    for c in range(3):
        np.maximum(
            frame[:, :, c], prev_frame[:, :, c],
            out=composite_frame[:, :, c]
        )

    luminance_frame = np.dot(composite_frame, LUM_COEFFS)

    out_frame = skimage.transform.resize(luminance_frame, out_size)
    out_frame /= 255.0

    return torch.from_numpy(out_frame).float()


class AtariEnvWrapper:
    """
    Wrapper around Atari games. Maintains API from gymnasium
    Implements frame averaging & stacking, action reptition, and reward clipping
    """

    def __init__(self, game_name, n_repeat=4, reward_clip=1.0, **kwargs):
        self.env = gym.make(game_name, obs_type='rgb', **kwargs)
        self.n_repeat = n_repeat
        self.frame_buffer = []
        self.reward_clip = reward_clip

    def step(self, action):
        reward_buffer = []
        for i in range(self.n_repeat):
            frame, reward, terminated, trunc, info = self.env.step(action)
            self.frame_buffer.append(frame)
            reward_buffer.append(
                min(max(reward, -self.reward_clip), self.reward_clip)
            )
            if terminated:
                break

        frames = []
        if not terminated:
            # Pre-process and stack frames for the model
            for i in range(self.n_repeat):
                frame = preprocess_frames(
                    self.frame_buffer[i], self.frame_buffer[i+1]
                )
                frames.append(frame)

            frames = torch.dstack(frames).permute(2, 0, 1)
            # Keep last frame to do the average next time
            self.frame_buffer = self.frame_buffer[-1:]

        # TODO: Is clipping done here or frame by frame?
        total_reward = sum(reward_buffer)

        return frames, total_reward, terminated, trunc, info

    def reset(self, seed=None):
        # Reset the env and save initial frame
        frame, info = self.env.reset(seed=seed)
        self.frame_buffer = [frame]
        # Repeat the frame N times to run through the model
        start_frame = preprocess_frames(frame, frame)
        frames = torch.dstack([start_frame]*self.n_repeat).permute(2, 0, 1)

        return frames, info

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def spec(self):
        return self.env.spec
