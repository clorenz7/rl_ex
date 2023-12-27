
import gymnasium as gym
import numpy as np
import skimage
import torch

LUM_COEFFS = np.array([0.299, 0.587, 0.114])


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

    def __init__(self, game_name, n_repeat=4, reward_clip=1.0, trim_x=0, **kwargs):
        self.env = gym.make(game_name, obs_type='rgb', frameskip=1, **kwargs)
        self.n_repeat = n_repeat
        self.frame_buffer = []
        self.reward_clip = reward_clip
        self.num_lives = 0
        self.trim_x = trim_x

    def step(self, action):
        reward_buffer = []
        for i in range(self.n_repeat):
            frame, reward, terminated, trunc, info = self.env.step(action)
            self.frame_buffer.append(frame)

            if self.reward_clip:
                reward = min(max(reward, -self.reward_clip), self.reward_clip)
            reward_buffer.append(reward)

            # Detect a lost life and end the episode
            current_lives = info.get('lives', 0)
            lost_a_life = current_lives != self.num_lives
            self.num_lives = current_lives
            if terminated:
                break
            if lost_a_life:
                # Start the next life to initialize the frame buffer
                frame, _, _, _, _ = self.env.step(action)
                self.frame_buffer = [frame]
                break

        frames = []
        if not (terminated or lost_a_life):
            # Pre-process and stack frames for the model
            for i in range(self.n_repeat):
                frame = preprocess_frames(
                    self.frame_buffer[i+1], self.frame_buffer[i],
                    trim_x=self.trim_x
                )
                frames.append(frame)

            frames = torch.dstack(frames).permute(2, 0, 1)
            # Keep last frame to do the average next time
            self.frame_buffer = self.frame_buffer[-1:]

        # TODO: Is clipping done here or frame by frame?
        # Hypothesis: can't score more than once in a small interval of frames.
        total_reward = sum(reward_buffer)

        return frames, total_reward, terminated, trunc, info

    def reset(self, seed=None):
        # Reset the env and save initial frame
        frame, info = self.env.reset(seed=seed)
        self.num_lives = info.get('lives', 0)
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
