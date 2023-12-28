from torch import nn
from .a2c import AdvantageActorCriticAgent


class Mnih2015PolicyValueNetwork(nn.Module):
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
            nn.Flatten(start_dim=0),  # Not doing batches
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(n_hidden, n_actions),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(n_hidden, 1)

    def forward(self, x):
        """
        x shape: n_frames x 84 x 84
        """
        x = self.base_net(x)
        action_probs = self.policy_head(x)
        value_est = self.value_head(x)

        return action_probs, value_est


class Mnih2016A2CAgent(AdvantageActorCriticAgent):

    def construct_net(self):
        return Mnih2015PolicyValueNetwork(self.n_actions)

    def normalize_state(self, state):
        return state

