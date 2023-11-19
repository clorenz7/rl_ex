"""
Mountain car
- Initial version is tile coding with linear function approx.
"""
import argparse
import json
import random
import time

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
            action_vals = self.net(features)
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
            ax.plot_surface(grid_x, grid_y, time_to_go); plt.show()

        return q_vals


class TorchAgentBase(MountainCarAgent):

    def __init__(self, n_actions, gamma=1.0, epsilon=1e-8,
                 min_vals=MIN_VALS, max_vals=MAX_VALS,
                 alpha=0.1/8, device="cpu"):
        super().__init__(
            n_actions, min_vals=min_vals, max_vals=max_vals,
            gamma=gamma, epsilon=epsilon, alpha=alpha,
        )
        self.device = device
        self.mu = (torch.tensor(max_vals) + torch.tensor(min_vals))/2.0
        self.sigma = (torch.tensor(max_vals) - torch.tensor(min_vals))/2.0

    def state_to_features(self, state):
        features = (
            (torch.tensor(state) - self.mu)/self.sigma
        ).to(self.device)

        return features

    def step(self, reward, state, debug=False):
        if debug:
            weights = self.net[0].weight.clone().detach()
        features = self.state_to_features(state)
        next_action = self.select_action(features=features)

        with torch.no_grad():
            self.net.eval()
            next_value = self.net(features)[next_action].item()
            if debug:
                last_value = self.net(self.last_features)[self.last_action]
            self.net.train()

        self.optimizer.zero_grad(set_to_none=True)
        delta = (
            reward + self.gamma * next_value
            - self.net(self.last_features)[self.last_action]
        )
        loss = delta ** 2
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

        return self.last_action

    def finish(self, reward):
        self.optimizer.zero_grad(set_to_none=True)
        last_features = self.state_to_features(self.last_state)

        loss = (
            reward - self.net(last_features)[self.last_action]
        )
        loss.backward()
        self.optimizer.step()

    def reset(self):
        self.net = nn.Linear(
            self.n_feats, self.n_actions, bias=False
        ).to(self.device)
        self.net.weight.data = self.net.weight.data/1000

        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            momentum=0.0,
            weight_decay=0.0,
            lr=self.alpha/2
        )


class TiledLinearAgent(TorchAgentBase):

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


class TiledLinearAgentSutton(TiledLinearAgent):
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


class FFWAgent(TorchAgentBase):

    def __init__(self, n_actions, n_hidden=8, n_layers=2,
                 min_vals=MIN_VALS, max_vals=MAX_VALS,
                 alpha=0.1/8, epsilon=0.1, device="cpu",
                 optimizer="adam", weight_decay=0, dropout=0):
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_state = 2
        super().__init__(
            n_actions, min_vals=min_vals, max_vals=max_vals,
            gamma=1.0, epsilon=epsilon, alpha=alpha, device=device
        )
        self.optimizer_name = optimizer.lower()
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.reset()

    def reset(self):
        layers = []
        last_out = self.n_state
        for ii in range(self.n_layers):
            is_last = ii == self.n_layers-1
            next_out = self.n_actions if is_last else self.n_hidden
            layer = nn.Linear(last_out, next_out, bias=True)
            # layer.weight.data = layer.weight.data/10
            layers.append(layer)
            if not is_last:
                layers.append(nn.ReLU())
                if self.dropout:
                    layers.append(nn.Dropout(p=self.dropout))
                # layers.append(nn.Sigmoid())
            last_out = self.n_hidden

        self.net = nn.Sequential(
            *layers
        ).to(self.device).train()

        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                momentum=0.0,
                weight_decay=0.0,
                lr=self.alpha
            )


class FFWAgent2(TorchAgentBase):

    def __init__(self, n_actions, agent_params={}, train_params={}, device="cpu"):
        self.n_hidden = agent_params.get('n_hidden', 512)
        self.n_layers = agent_params.get('n_layers', 3)
        self.n_state = 2
        min_vals = agent_params.get('min_vals', MIN_VALS)
        max_vals = agent_params.get('max_vals', MIN_VALS)

        gamma = agent_params.get('gamma', 1.0)
        epsilon = agent_params.get('epsilon', 1e-2)
        alpha = agent_params.get('alpha', 1e-6)

        self.train_params = train_params
        self.optimizer_name = self.train_params.pop('optimizer', 'adam').lower()
        alpha = self.train_params.pop('alpha', None)
        if alpha is not None:
            self.train_params['lr'] = alpha

        super().__init__(
            n_actions, min_vals=min_vals, max_vals=max_vals,
            gamma=gamma, epsilon=epsilon, alpha=alpha, device=device
        )
        self.reset()

    def reset(self):
        layers = []
        last_out = self.n_state
        for ii in range(self.n_layers):
            is_last = ii == self.n_layers-1
            next_out = self.n_actions if is_last else self.n_hidden
            layer = nn.Linear(last_out, next_out, bias=True)
            layers.append(layer)
            if not is_last:
                layers.append(nn.ReLU())
            last_out = self.n_hidden

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

def experiment_loop(env, agent, seed=101, n_runs=100, n_episodes=500,
                    eval_episodes={}, max_steps=10000, verbose=True):

    # actions = list(range(env.action_space.n))
    run_steps = np.zeros((n_episodes, n_runs))

    state, info = env.reset(seed=seed)
    for r_idx in range(n_runs):
        for e_idx in range(n_episodes):
            s_idx = 0
            action_idx = agent.initialize(state)
            trajectory = [state]
            actions = []
            st = time.time()
            state, reward, terminated, _, _ = env.step(action_idx)
            while not terminated:
                action_idx = agent.step(reward, state)
                state, reward, terminated, _, _ = env.step(action_idx)
                trajectory.append(state)
                s_idx += 1
                actions.append(action_idx)

                # Pause if loop might be infinite
                if s_idx > max_steps:
                    elap = round((time.time() - st)/60, 2)
                    # print(f"Break at {s_idx} after {elap}min")
                    # agent.evaluate_q()
                    # np_traj = np.vstack(trajectory).T
                    # import ipdb; ipdb.set_trace()
                    break
            if terminated:
                agent.finish(reward)
            elap = round((time.time() - st)/60, 2)
            # agent.evaluate_q()
            # Record result and display if desired
            run_steps[e_idx, r_idx] = s_idx
            if verbose or (e_idx + 1) % 100 == 0:
                print(f"Run {r_idx+1} Episode {e_idx+1} terminated at step {s_idx} in {elap}min")
            if e_idx in eval_episodes:
                q_vals = agent.evaluate_q()

            state, info = env.reset()
        agent.reset()

    return run_steps


class ExperimentParams:

    def __init__(self, agent_params={}, train_params={},
                 simulation_params={}):
        self.agent_params = agent_params
        self.train_params = train_params
        self.simulation_params = simulation_params


def main():

    # parser = argparse.ArgumentParser(
    #     description="Train an RL Agent for Mountain Car"
    # )
    # parser.add_argument(
    #     '-j', "--json", type=str, default="",
    #     help="Json experiment parameters"
    # )

    # cli_args = parser.parse_args()

    # with open(cli_args.json, 'r') as fp:
    #     json_params = json.load(fp)
    #     exp_params = ExperimentParams(**json_params)

    env = gym.make('MountainCar-v0')
    n_actions = env.action_space.n
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = "cpu"

    # agent = TiledLinearAgent(n_actions, alpha=0.5/8)
    # agent.evaluate_q()
    # run_steps = experiment_loop(env, agent, n_episodes=200, n_runs=20)

    # This sorta worked with SGD (first took 7976 steps):
    # agent = FFWAgent(
    #     n_actions, alpha=1e-3, device=device, n_hidden=512, n_layers=4,
    #     epsilon=0.1
    # )
    # This kinda works with Adam:
    # agent = FFWAgent(
    #     n_actions, alpha=1e-5, device=device, n_hidden=512, n_layers=4,
    #     epsilon=0.05, optimizer="adam"
    # )

    # agent = FFWAgent(
    #     n_actions, alpha=1e-3, device=device, n_hidden=1024, n_layers=4,
    #     epsilon=0.05
    # )

    # epislon of 0.01 seems to be better than 0.05... at alpha of 1e-5
    # lr of 6e-6 seeems to be best so far.
    # agent = FFWAgent(
    #     n_actions, alpha=6e-6, device=device, n_hidden=512, n_layers=4,
    #     epsilon=1e-4, optimizer="adam"
    # )
    # This starts well and then degrades: (weight decay needed?)
    # agent = FFWAgent(
    #     n_actions, alpha=1e-5, device=device, n_hidden=512, n_layers=3,
    #     epsilon=1e-4, optimizer="adam"
    # )
    # Weight decay sort of helps (1e-4 initially)
    # agent = FFWAgent(
    #     n_actions, alpha=1e-5, device=device, n_hidden=1024, n_layers=3,
    #     epsilon=1e-4, optimizer="adam", weight_decay=1e-3
    # )

    # # This did quite a bit better
    # agent = FFWAgent(
    #     n_actions, alpha=2e-6, device=device, n_hidden=1024, n_layers=4,
    #     epsilon=1e-4, optimizer="adam", weight_decay=1e-3
    # )
    # run_steps = experiment_loop(env, agent, n_episodes=100, n_runs=10, max_steps=15000)

    # q_vals = agent.evaluate_q()
    # This did not over train:
    # agent = FFWAgent(
    #     n_actions, alpha=5e-7, device=device, n_hidden=1024, n_layers=4,
    #     epsilon=1e-3, optimizer="adam", weight_decay=1e-2
    # )
    # run_steps = experiment_loop(env, agent, n_episodes=80, n_runs=10, max_steps=14000)

    agent = FFWAgent(
        n_actions, alpha=2e-6, device=device, n_hidden=1024, n_layers=4,
        epsilon=1e-3, optimizer="adam", weight_decay=1e-2
    )
    run_steps = experiment_loop(env, agent, n_episodes=250, n_runs=10, max_steps=14000)



    avg_steps = np.mean(run_steps, axis=1)
    plt.semilogy(avg_steps);
    plt.xlabel('Episode #');
    plt.ylim(bottom=100);
    plt.ylabel('Avg # of Steps to Reach Goal');
    plt.show()

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
