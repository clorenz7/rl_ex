
import numpy as np

import skimage
from torch import nn

LUM_COEFFS = np.array([0.299, 0.587, 0.114])


class PolicyValueImageNetwork(nn.Module):
    def __init__(self, image_size, n_actions):
        # Image size: 84 x 84 x 1
        n_channels = 1  # Might be 4 frames stacked together?
        n_filters_1 = 16
        filter_size_1 = (8, 8)
        stride_1 = 4

        n_filters_2 = 32
        filter_size_2 = (4, 4)
        stride_2 = 2

        n_features = 8 * 8
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

    return out_frame
