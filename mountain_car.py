"""
Mountain car
- Initial version is tile coding with linear function approx.
"""
import random

import gymnasium as gym
import matplotlib.pyplot as plt
from mushroom_rl.features.tiles import Tiles
import numpy as np
import torch
from torch import nn

import sutton_tiles


MIN_VALS = [-1.2, -0.07]
MAX_VALS = [0.6, 0.07]


class MountainCarAgent:

    def __init__(self, n_actions, min_vals=MIN_VALS, max_vals=MAX_VALS,
                 gamma=1.0, epsilon=1e-8, alpha=None):
        self.last_state = None
        self.last_action = None
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.alpha = alpha or 0.1/8

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

        with torch.no_grad():
            action_vals = self.layer(features)
            if random.random() >= self.epsilon:
                action_idx = int(torch.argmax(action_vals))
            else:
                action_idx = random.randint(0, self.n_actions-1)

        return action_idx

    def initialize(self, state):
        self.last_state = state
        self.last_features = self.state_to_features(state)
        self.last_action = self.select_action(state)
        return self.last_action

    def evaluate_q(self):
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
            q_vals = self.layer(features)
        q_vals = q_vals.reshape([n_mesh, n_mesh, 3]).detach().numpy()
        time_to_go = -np.max(q_vals, axis=2)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(grid_x, grid_y, time_to_go); plt.show()


class TiledLinearAgent(MountainCarAgent):

    def __init__(self, n_actions, n_grid=8, n_tiles=8,
                 min_vals=MIN_VALS, max_vals=MAX_VALS,
                 alpha=0.1/8):
        super().__init__(
            n_actions, min_vals=min_vals, max_vals=max_vals,
            gamma=1.0, epsilon=1e-8, alpha=alpha,
        )
        self.n_grid = n_grid
        self.n_tiles = n_tiles

        self.tiling = Tiles.generate(
            n_tiles, [n_grid, n_grid],
            min_vals, max_vals,
            uniform=True
        )
        self.n_grid_entries = n_grid * n_grid
        self.n_feats = n_grid * n_grid * n_tiles

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

    def step(self, reward, state, debug=False):
        weights = self.layer.weight.clone().detach()
        features = self.state_to_features(state)
        next_action = self.select_action(features=features)

        with torch.no_grad():
            next_value = self.layer(features)[next_action].item()
            if debug:
                last_value = self.layer(self.last_features)[self.last_action]

        self.optimizer.zero_grad(set_to_none=True)
        delta = (
            reward + self.gamma * next_value
            - self.layer(self.last_features)[self.last_action]
        )
        loss = delta ** 2
        if debug:
            loss.retain_grad()
        loss.backward()
        self.optimizer.step()

        if debug:
            # This is sanity checking.
            grad = self.last_features
            actual_update = self.layer.weight.detach() - weights
            exp_update = 2 * self.optimizer.param_groups[0]['lr'] * delta.clone().detach() * grad

            with torch.no_grad():
                updated_value = self.layer(self.last_features)[self.last_action]

            value_delta = updated_value - last_value
            weight_delta = torch.abs(actual_update[self.last_action, :] - exp_update).max()

        self.last_state = state
        self.last_features = features
        self.last_action = next_action

        return self.last_action

    def finish(self, reward):
        self.optimizer.zero_grad(set_to_none=True)
        last_features = self.state_to_features(self.last_state)

        loss = (
            reward - self.layer(last_features)[self.last_action]
        )
        loss.backward()
        self.optimizer.step()

    def reset(self):
        self.layer = nn.Linear(
            self.n_feats, self.n_actions, bias=False
        )
        self.layer.weight.data = self.layer.weight.data/1000

        self.optimizer = torch.optim.SGD(
            self.layer.parameters(),
            momentum=0.0,
            weight_decay=0.0,
            lr=self.alpha/2
        )


class TiledLinearAgent2(TiledLinearAgent):
    """
    Work in Progress class to use Sutton's tiling
    """
    def __init__(self, n_actions, n_grid=8, n_tiles=8,
                 min_vals=MIN_VALS, max_vals=MAX_VALS,
                 alpha=0.1/8):
        super().__init__(
            n_actions, min_vals=min_vals, max_vals=max_vals,
            gamma=1.0, epsilon=1e-8
        )
        self.n_grid = n_grid
        self.n_tiles = n_tiles
        self.alpha = alpha

        self.iht = sutton_tiles.IHT(4096)

        self.reset()

    def state_to_features(self, state, action):
        features = torch.zeros(self.iht.size)
        x = state[0]
        x_dot = state[1]
        indices = sutton_tiles.tiles(
            self.iht, 8, [8*x/1.7, 8*x_dot/.14], [action]
        )
        features[indices] = 1.0

        return features


def experiment_loop(env, agent, seed=101, n_runs=100, n_episodes=500,
                    eval_episodes={}):

    actions = list(range(env.action_space.n))
    run_steps = np.zeros((n_episodes, n_runs))

    state, info = env.reset(seed=seed)
    for r_idx in range(n_runs):
        for e_idx in range(n_episodes):
            s_idx = 0
            action_idx = agent.initialize(state)
            trajectory = [state]
            state, reward, terminated, _, _ = env.step(actions[action_idx])
            while not terminated:
                action_idx = agent.step(reward, state)
                state, reward, terminated, _, _ = env.step(actions[action_idx])
                trajectory.append(state)
                s_idx += 1

                # Pause if loop might be infinite
                if s_idx > 100000:
                    np_traj = np.vstack(trajectory)
                    import ipdb; ipdb.set_trace()
            agent.finish(reward)

            # Record result and display if desired
            run_steps[e_idx, r_idx] = s_idx
            if (e_idx + 1) % 100 == 0:
                print(f"Run {r_idx+1} Episode {e_idx+1} terminated at step {s_idx}")
            if e_idx in eval_episodes:
                q_vals = agent.evaluate_q()

            state, info = env.reset()
        agent.reset()

    return run_steps


def main():
    env = gym.make('MountainCar-v0')
    n_actions = env.action_space.n
    agent = TiledLinearAgent(n_actions, alpha=0.5/8)
    run_steps = experiment_loop(env, agent, n_episodes=200, n_runs=20)

    avg_steps = np.mean(run_steps, axis=1)
    plt.semilogy(avg_steps); plt.show()

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
