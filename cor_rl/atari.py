
import gymnasium as gym
import numpy as np
import skimage
import torch

LUM_COEFFS = np.array([0.299, 0.587, 0.114])
NO_OP = 0


def preprocess_frames(frame, prev_frame, trim_x=0, out_size=(84, 84)):
    composite_frame = np.zeros(frame.shape, dtype=frame.dtype)
    for c in range(3):
        np.maximum(
            frame[:, :, c], prev_frame[:, :, c],
            out=composite_frame[:, :, c]
        )
    luminance_frame = np.dot(composite_frame, LUM_COEFFS)

    if trim_x:
        luminance_frame = luminance_frame[:, trim_x:-trim_x]

    out_frame = skimage.transform.resize(luminance_frame, out_size)
    out_frame /= 255.0

    return torch.from_numpy(out_frame).float()


class AtariEnvWrapper:
    """
    Wrapper around Atari games. Maintains API from gymnasium
    Implements frame averaging & stacking, action reptition, and reward clipping
    """

    def __init__(self, game_name, n_stack=4, n_repeat=4, reward_clip=None,
                 trim_x=0, noop_max=30, lost_life_ends_ep=False, **kwargs):
        self.env = gym.make(game_name, obs_type='rgb', frameskip=1, **kwargs)
        self.n_repeat = n_repeat
        self.n_stack = n_stack
        self.frame_buffer = []
        self.reward_clip = reward_clip
        self.num_lives = 0
        self.trim_x = trim_x
        self.noop_max = noop_max
        self.lost_life_ends_ep = lost_life_ends_ep
        self.raw_frame_buffer = []

    def reload_frame_buffer(self, frame_0=None, frame_1=None):
        if frame_0 is None:
            frame_0, _, _, _, _ = self.env.step(NO_OP)
        if frame_1 is None:
            frame_1, _, _, _, _ = self.env.step(NO_OP)
        frame = preprocess_frames(
            frame_1, frame_0,
            trim_x=self.trim_x
        )
        self.frame_buffer = [frame] * self.n_stack

    def step(self, action):
        lost_a_life = False
        reward_buffer = []
        self.raw_frame_buffer = []
        for i in range(self.n_repeat):
            frame, reward, terminated, trunc, info = self.env.step(action)
            self.raw_frame_buffer.append(frame)

            if self.reward_clip:
                reward = min(max(reward, -self.reward_clip), self.reward_clip)
            reward_buffer.append(reward)

            # Detect a lost life and end the episode
            current_lives = info.get('lives', 0)
            if self.lost_life_ends_ep:
                lost_a_life = current_lives != self.num_lives
            self.num_lives = current_lives
            if terminated:
                break
            if lost_a_life:
                self.reload_frame_buffer(frame_0=frame, frame_1=frame)
                break

        if terminated or lost_a_life:
            frames = None
        else:
            frame = preprocess_frames(
                self.raw_frame_buffer[-1], self.raw_frame_buffer[-2],
                trim_x=self.trim_x
            )
            self.frame_buffer.append(frame)
            self.frame_buffer = self.frame_buffer[-self.n_stack:]

            frames = torch.dstack(self.frame_buffer).permute(2, 0, 1)

        total_reward = sum(reward_buffer)

        return frames, total_reward, terminated, trunc, info

    def render(self):
        if self.raw_frame_buffer:
            frame = 0
            for f in self.raw_frame_buffer:
                frame += f.astype(np.uint16)
            frame = frame / len(self.raw_frame_buffer)
            return frame.astype(np.uint8)
        else:
            return self.env.render()

    def reset(self, seed=None):
        # Reset the env and save initial frame
        frame, info = self.env.reset(seed=seed)
        self.reload_frame_buffer(frame_0=frame)
        self.num_lives = info.get('lives', 0)

        n_no_ops = self.env.np_random.integers(1, self.noop_max+1)

        for _ in range(n_no_ops):
            frames, total_reward, terminated, trunc, info = self.step(NO_OP)

        return frames, info

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def spec(self):
        return self.env.spec


class FrameStacker:
    def __init__(self, env, n_stack=4):
        self.env = env
        self.n_stack = n_stack
        self.frame_buffer = []

    def step(self, action):
        frame, reward, terminated, trunc, info = self.env.step(action)
        self.frame_buffer.append(torch.from_numpy(frame))
        self.frame_buffer = self.frame_buffer[-self.n_stack:]

        frames = torch.dstack(self.frame_buffer).permute(2, 0, 1)

        return frames, reward, terminated, trunc, info

    def reset(self, seed=None):
        frame, info = self.env.reset(seed=seed)
        self.frame_buffer = [torch.from_numpy(frame)] * self.n_stack

        frames = torch.dstack(self.frame_buffer).permute(2, 0, 1)

        return frames, info
