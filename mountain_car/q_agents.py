import random

import numpy as np
import matplotlib.pyplot as plt
from mushroom_rl.features.tiles import Tiles
import torch
from torch import nn
import torch.nn.functional as F

import sutton_tiles

MIN_VALS = [-1.2, -0.07]
MAX_VALS = [0.6, 0.07]


class ActivationFactory:
    _MAP = {
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'hardtanh': nn.Hardtanh,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'hardsigmoid': nn.Hardsigmoid,
    }

    def get(self, activation_name):
        return self._MAP[activation_name.lower()]()


activation_factory = ActivationFactory()


class TorchQAgentBase:

    def __init__(self, n_actions, agent_params={}, train_params={}, device="",
                 n_state=2):
        self.n_actions = n_actions
        self.n_state = n_state
        self.last_state = None
        self.last_action = None

        self.agent_params = agent_params or {}
        self.train_params = train_params or {}

        self.gamma = self.agent_params.get('gamma', 1.0)
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

        self.train_params = train_params
        self.optimizer_name = self.train_params.pop('optimizer', 'adam').lower()
        alpha = self.train_params.pop('alpha', None)
        if alpha is not None:
            self.train_params['lr'] = alpha

        self.device = device or "cpu"

    def bound(self, state, eps=1e-4):
        for i in range(len(state)):
            state[i] = max(
                min(state[i], self.max_vals[i]-eps),
                self.min_vals[i] + eps
            )
        return state

    def select_action(self, state=None, features=None):
        if state is not None:
            features = self.state_to_features(state)

        action_vals = self.net(features)
        action_idx = int(torch.argmax(action_vals))
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.n_actions-1)
        action_value = action_vals[action_idx]

        return action_idx, action_value

    def initialize(self, state):
        self.optimizer.zero_grad(set_to_none=True)
        self.last_state = state
        features = self.state_to_features(state)
        self.last_action, action_value = self.select_action(features=features)
        self.last_features = features
        self.last_action_value = action_value

        return self.last_action

    def evaluate_q(self, show_v=True):
        n_mesh = 100
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
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_xf, grid_yf = grid_x.flatten(), grid_y.flatten()
        features = []
        for state in zip(grid_xf, grid_yf):
            features.append(self.state_to_features(state))

        features = torch.vstack(features)

        with torch.no_grad():
            q_vals = self.net(features)
        q_vals = q_vals.reshape([n_mesh, n_mesh, 3]).detach().cpu().numpy()
        if show_v:
            time_to_go = -np.max(q_vals, axis=2)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(grid_x, grid_y, time_to_go);
            ax.set_xlabel("Position")
            ax.set_ylabel("Velocity")
            ax.set_zlabel("-V(s): steps to goal")
            plt.show()

        return q_vals

    def state_to_features(self, state):
        features = (
            (torch.tensor(state) - self.mu)/self.sigma
        ).to(self.device)

        return features

    def step(self, reward, state, debug=False):
        if debug:
            weights = self.net[0].weight.clone().detach()
        features = self.state_to_features(state)

        with torch.no_grad():
            # Empirically found that doing action selection here was much more stable
            next_action, next_action_value = self.select_action(features=features)

            self.net.eval()
            next_state_value = self.net(features).max().detach()
            if debug:
                last_value = self.net(self.last_features)[self.last_action]
            self.net.train()

        self.optimizer.zero_grad(set_to_none=True)
        last_action_value = self.net(self.last_features)[self.last_action]

        if not self.use_smooth_l1_loss:
            delta = (
                (reward + self.gamma * next_state_value) -
                last_action_value
            )
            loss = delta ** 2
        else:
            loss = F.smooth_l1_loss(
                reward + self.gamma * next_state_value,
                last_action_value
            )
        if debug:
            loss.retain_grad()
        loss.backward()
        self.optimizer.step()

        if debug:
            # This is sanity checking.
            grad = self.last_features
            actual_update = self.net[0].weight.detach() - weights
            exp_update = 2 * self.optimizer.param_groups[0]['lr'] * delta.clone().detach() * grad

            with torch.no_grad():
                updated_value = self.net(self.last_features)[self.last_action]

            value_delta = updated_value - last_value
            weight_delta = torch.abs(actual_update[self.last_action, :] - exp_update).max()

        self.last_state = state
        self.last_features = features
        self.last_action = next_action
        self.last_action_value = next_action_value

        return self.last_action

    def finish(self, reward):

        self.optimizer.zero_grad(set_to_none=True)
        last_action_value = self.net(self.last_features)[self.last_action]

        if not self.use_smooth_l1_loss:
            loss = (reward - last_action_value)**2
        else:
            loss = F.smooth_l1_loss(
                torch.tensor(reward, device=self.device),
                last_action_value
            )
        loss.backward()
        self.optimizer.step()

    def checkpoint(self, file_name):
        torch.save(self.net, file_name)


class TiledLinearQAgent(TorchQAgentBase):

    def __init__(self, n_actions, agent_params={}, train_params={}, device="cpu"):
        super().__init__(n_actions, agent_params, train_params, device)

        self.n_grid = self.agent_params.get('n_grid', 8)
        self.n_tiles = self.agent_params.get('n_tiles', 8)

        self.tiling = Tiles.generate(
            self.n_tiles, [self.n_grid, self.n_grid],
            self.min_vals, self.max_vals,
            uniform=True
        )
        self.n_grid_entries = self.n_grid * self.n_grid
        self.n_feats = self.n_grid * self.n_grid * self.n_tiles

        self.reset()

    def state_to_features(self, state):
        features = torch.zeros(self.n_feats)
        for i in range(self.n_tiles):
            index = self.tiling[i](state)
            if index is None:
                state = self.bound(state)
                index = self.tiling[i](state)
            features[i * self.n_grid_entries + index] = 1

        return features

    def reset(self):
        self.net = nn.Linear(
            self.n_feats, self.n_actions, bias=False
        ).to(self.device)
        self.net.weight.data = self.net.weight.data/1000

        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            momentum=0.0,
            weight_decay=0.0,
            lr=self.train_params['lr']/2.0
        )


class TiledLinearQAgentSutton(TiledLinearQAgent):
    """
    Work in Progress class to use Sutton's tiling
    """
    def __init__(self, n_actions, agent_params={}, train_params={}, device="cpu"):
        super().__init__(n_actions, agent_params, train_params, device)

        self.n_grid = self.agent_params.get('n_grid', 8)
        self.n_tiles = self.agent_params.get('n_tiles', 8)

        self.iht = sutton_tiles.IHT(4096)

        self.scales = (self.n_grid / (2.0 * self.sigma)).tolist()

        self.reset()

    def state_to_features(self, state, action):
        features = torch.zeros(self.iht.size, device=self.device)
        indices = sutton_tiles.tiles(
            self.iht, self.n_tiles,
            [state[0] * self.scales[0], state[1] * self.scales[1]],
            [action]
        )
        features[indices] = 1.0

        return features


class FFWQAgent(TorchQAgentBase):

    def __init__(self, n_actions, agent_params={}, train_params={}, device="cpu"):
        super().__init__(n_actions, agent_params, train_params, device)

        n_hidden = agent_params.get('n_hidden', 512)
        if isinstance(n_hidden, list):
            self.n_hidden = n_hidden
            self.n_layers = len(n_hidden) + 1
        else:
            self.n_layers = agent_params.get('n_layers', 3)
            self.n_hidden = [n_hidden] * self.n_layers

        self.activation_name = agent_params.get('activation', 'elu')

        self.reset()

    def reset(self):
        layers = []
        last_out = self.n_state
        for ii in range(self.n_layers):
            is_not_last = ii != self.n_layers-1
            next_out = self.n_hidden[ii] if is_not_last else self.n_actions
            layers.append(
                nn.Linear(last_out, next_out, bias=True)
            )
            if is_not_last:
                layers.append(
                    activation_factory.get(self.activation_name)
                )
            last_out = next_out

        self.net = nn.Sequential(
            *layers
        ).to(self.device)

        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(),
                **self.train_params
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                **self.train_params
            )
