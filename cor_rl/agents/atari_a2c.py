from torch import nn
from .a2c import AdvantageActorCriticAgent


class Mnih2015PolicyValueNetwork(nn.Module):
    def __init__(self, n_actions, n_channels=4, n_hidden=256):
        super().__init__()
        # Image size: 84 x 84 x n_channels
        n_filters_1 = 16
        filter_size_1 = (8, 8)
        stride_1 = 4

        n_filters_2 = 32
        filter_size_2 = (4, 4)
        stride_2 = 2

        n_features = 9 * 9 * n_filters_2

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

        return action_probs, value_est, None


class KostikovPVLSTMNetwork(nn.Module):
    """
    Based on https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py
    """

    def __init__(self, n_actions, n_channels=1, n_recurrent=256):
        super().__init__()
        # Image size: 84 x 84 x n_channels
        stride = 2
        n_filters = 32
        kernel_size = 3
        n_flattened = 3 * 3 * 32

        self.base_net = nn.Sequential(
            # Added this extra layer to take 84 x 84 down to 42x42.
            nn.Conv2d(n_channels, n_filters, kernel_size, stride, padding=1),
            nn.ELU(),
            nn.Conv2d(n_filters, n_filters, kernel_size, stride, padding=1),  # 21 x 21
            nn.ELU(),
            nn.Conv2d(n_filters, n_filters, kernel_size, stride, padding=1),  # 11 x 11
            nn.ELU(),
            nn.Conv2d(n_filters, n_filters, kernel_size, stride, padding=1),  # 6 x 6
            nn.ELU(),
            nn.Conv2d(n_filters, n_filters, kernel_size, stride, padding=1),  # 3 x 3
            nn.ELU(),
            # nn.Conv2d(n_filters, n_filters, kernel_size, stride, padding=1),
            # nn.ELU(),
            nn.Flatten(start_dim=0),
        )

        self.lstm = nn.LSTMCell(n_flattened, n_recurrent)

        self.policy_head = nn.Sequential(
            nn.Linear(n_recurrent, n_actions),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(n_recurrent, 1)

    def forward(self, state):
        x, recurrent_state = state
        x = self.base_net(x)
        new_recurrent_state = self.lstm(x, recurrent_state)

        x = new_recurrent_state[0]
        action_probs = self.policy_head(x)
        value_est = self.value_head(x)

        return action_probs, value_est, new_recurrent_state


class Mnih2016PolicyValueLSTMNetwork(Mnih2015PolicyValueNetwork):
    def __init__(self, n_actions, n_channels=4, n_hidden=256):
        super().__init__(n_actions, n_channels, n_hidden)

        self.lstm = nn.LSTMCell(n_hidden, n_hidden)

    def forward(self, state):
        x, recurrent_state = state
        x = self.base_net(x)
        new_recurrent_state = self.lstm(x, recurrent_state)

        x = new_recurrent_state[0]
        action_probs = self.policy_head(x)
        value_est = self.value_head(x)

        return action_probs, value_est, new_recurrent_state


class Mnih2016A2CAgent(AdvantageActorCriticAgent):

    def construct_net(self):
        return Mnih2015PolicyValueNetwork(self.n_actions)


class Mnih2016LSTMA2CAgent(AdvantageActorCriticAgent):

    n_channels = 4

    def construct_net(self):
        return Mnih2016PolicyValueLSTMNetwork(self.n_actions, self.n_channels)


class KostikovLSTMA2CAgent(AdvantageActorCriticAgent):
    def construct_net(self):
        return KostikovPVLSTMNetwork(self.n_actions, self.n_channels)