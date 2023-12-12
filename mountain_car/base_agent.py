
import torch
from torch import nn

MIN_VALS = [-1.2, -0.07]
MAX_VALS = [0.6, 0.07]


class BaseAgent:

    def __init__(self, agent_params):
        self.agent_params = agent_params or {}
        self.gamma = self.agent_params.get('gamma', 0.9)
        self.epsilon = self.agent_params.get('epsilon', 1e-8)
        self.min_vals = self.agent_params.get('min_vals', MIN_VALS)
        self.max_vals = self.agent_params.get('max_vals', MAX_VALS)
        self.use_smooth_l1_loss = self.agent_params.get('use_smooth_l1_loss', False)
        self.mu = (
            torch.tensor(self.max_vals) + torch.tensor(self.min_vals)
        ) / 2.0
        self.sigma = (
            torch.tensor(self.max_vals) - torch.tensor(self.min_vals)
        ) / 2.0

    def get_grid(self, n_mesh=100):
        x = torch.linspace(
            self.min_vals[0] + 1e-3,
            self.max_vals[0] - 1e-3,
            n_mesh
        )
        y = torch.linspace(
            self.min_vals[1] + 1e-3,
            self.max_vals[1] - 1e-3,
            n_mesh
        )
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        grid_xf, grid_yf = grid_x.flatten(), grid_y.flatten()
        features = []
        for state in zip(grid_xf, grid_yf):
            features.append(self.state_to_features(state))

        features = torch.vstack(features).to(self.device)

        return features, grid_x.numpy(), grid_y.numpy()

    def bound(self, state, eps=1e-4):
        for i in range(len(state)):
            state[i] = max(
                min(state[i], self.max_vals[i]-eps),
                self.min_vals[i] + eps
            )
        return state

    def normalize_state(self, state):
        features = (
            (torch.tensor(state) - self.mu)/self.sigma
        ).to(self.device)

        return features


class ActivationFactory:
    _MAP = {
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'elu': nn.ELU,
        'hardtanh': nn.Hardtanh,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'hardsigmoid': nn.Hardsigmoid,
    }

    def get(self, activation_name):
        return self._MAP[activation_name.lower()]()


activation_factory = ActivationFactory()


def net_from_layer_sizes(layer_sizes, activation=nn.ELU, final_activation=None):

    if isinstance(activation, str):
        activation = activation_factory.get(activation)

    if isinstance(final_activation, str):
        final_activation = activation_factory.get(final_activation)

    layers = []
    for ii in range(len(layer_sizes)-1):
        layers.append(nn.Linear(layer_sizes[ii], layer_sizes[ii+1]))
        layers.append(activation())

    # Remove final activation and re-add it
    layers.pop()
    if final_activation is not None:
        layers.append(final_activation())

    return nn.Sequential(*layers)
