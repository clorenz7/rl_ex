from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from base_agent import BaseAgent, net_from_layer_sizes


class PolicyValueNetwork(nn.Module):

    def __init__(self, n_actions, n_hidden, n_state=2):
        super().__init__()
        self.base_layer = nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(n_hidden, n_actions)
        self.value_head = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.base_layer(x)

        logits = self.policy_head(x)

        action_probs = F.softmax(logits, dim=-1)

        value_est = self.value_head(x)

        return action_probs, value_est


class MountainCarActorCriticAgent(BaseAgent):

    def __init__(self, n_actions, agent_params={}, train_params={}, device="cpu"):
        super().__init__(agent_params)
        self.last_state = None
        self.last_action = None

        self.device = device
        self.n_actions = n_actions
        self.n_hidden = agent_params.get('n_hidden', None)

        self.actor_layers = agent_params.get('actor_layers', [48, 48])
        self.critic_layers = agent_params.get('critic_layers', [384, 384])

        self.n_state = agent_params.get('n_state', 2)
        self.train_params = train_params
        self.optimizer_name = self.train_params.pop('optimizer', 'adam').lower()

        alpha = self.train_params.pop('alpha', None)
        if alpha is not None:
            self.train_params['lr'] = alpha

        self.reset()

    def get_action_and_value(self, state=None, features=None):
        if state is not None:
            features = self.state_to_features(state)

        action_probs, value_est = self.net(features)

        pdf = Categorical(action_probs)
        action = pdf.sample()
        self.last_log_prob = pdf.log_prob(action)

        return action.item(), value_est

    def select_action(self, state=None, features=None):
        if state is not None:
            features = self.state_to_features(state)

        action_probs = self.actor(features)

        pdf = Categorical(action_probs)
        action = pdf.sample()
        self.last_log_prob = pdf.log_prob(action)

        return action.item()

    def initialize(self, state):
        self.last_state = state
        self.last_features = self.state_to_features(state)
        self.last_action = self.select_action(features=self.last_features)
        value_est = self.critic(self.last_features)
        self.last_value_est = value_est
        return self.last_action

    def state_to_features(self, state):
        return self.normalize_state(state)

    def step(self, reward, state, debug=False):

        features = self.state_to_features(state)

        last_log_prob = self.last_log_prob
        prev_value_est = self.last_value_est

        with torch.no_grad():
            value_est = self.critic(features)
        total_return_est = reward + self.gamma * value_est

        # Semi-gradient update
        delta = total_return_est - prev_value_est.item()

        policy_loss = delta * -last_log_prob
        value_loss = F.smooth_l1_loss(total_return_est, prev_value_est)
        loss = policy_loss + value_loss

        if self.optimizer is None:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            value_loss.backward(retain_graph=True)
            policy_loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step()
        else:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        next_action = self.select_action(features=features)
        value_est = self.critic(features)

        self.last_state = state
        self.last_features = features
        self.last_action = next_action
        self.last_value_est = value_est

        return next_action

    def finish(self, reward):
        last_log_prob = self.last_log_prob
        prev_value_est = self.last_value_est
        # Semi-gradient update
        delta = reward - prev_value_est.item()

        policy_loss = delta * -last_log_prob
        value_loss = F.smooth_l1_loss(
            torch.tensor([reward], device=self.device), prev_value_est
        )
        if self.optimizer is None:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            value_loss.backward(retain_graph=True)
            policy_loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step()
        else:
            self.optimizer.zero_grad(set_to_none=True)
            loss = policy_loss + value_loss
            loss.backward()
            self.optimizer.step()

    def set_optimizer(self):
        if self.n_hidden is not None:
            params = list(self.actor.parameters()) + list(self.critic.parameters())
            if self.optimizer_name == "adam":
                self.optimizer = torch.optim.AdamW(
                    params,
                    **self.train_params
                )
            else:
                self.optimizer = torch.optim.SGD(
                    params,
                    **self.train_params
                )
        else:
            self.optimizer = None
            self.actor_optimizer = torch.optim.AdamW(
                self.actor.parameters(),
                **self.train_params['actor']
            )
            self.critic_optimizer = torch.optim.AdamW(
                self.critic.parameters(),
                **self.train_params.get('critic', {})
            )

    def reset(self):

        if self.n_hidden is not None:
            self.actor = nn.Sequential(
                nn.Linear(self.n_state, self.n_hidden),
                nn.ELU(),
                nn.Linear(self.n_hidden, self.n_actions),
                nn.Softmax(dim=-1)
            ).to(self.device)
            self.critic = nn.Sequential(
                nn.Linear(self.n_state, self.n_hidden),
                nn.ELU(),
                nn.Linear(self.n_hidden, 1),
            ).to(self.device)

        else:
            actor_layer_sizes = [self.n_state] + self.actor_layers + [self.n_actions]
            self.actor = net_from_layer_sizes(
                actor_layer_sizes,
                final_activation=partial(nn.Softmax, dim=-1)
            ).to(self.device)
            critic_layer_sizes = [self.n_state] + self.critic_layers + [1]
            self.critic = net_from_layer_sizes(
                critic_layer_sizes,
            ).to(self.device)

        self.set_optimizer()

    def checkpoint(self, file_name):
        torch.save(self.actor, file_name + ".actor")
        torch.save(self.critic, file_name + ".critic")

    def load(self, actor_critic_base: str):
        actor_file = actor_critic_base + ".actor"
        critic_file = actor_critic_base + ".critic"
        self.actor = torch.load(actor_file).to(self.device)
        self.critic = torch.load(critic_file).to(self.device)

    def visualize(self):
        features, grid_x, grid_y = self.get_grid()
        n_mesh = grid_x.shape[0]
        with torch.no_grad():
            action_probs = self.actor(features).reshape([n_mesh, n_mesh, 3]).cpu().numpy()
            v_est = self.critic(features).reshape([n_mesh, n_mesh]).cpu().numpy()

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(grid_x, grid_y, -v_est);
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_zlabel("-V(s): steps to goal")

        plt.figure(2)
        tick_idx = np.linspace(1, grid_x.shape[0], 6, dtype=int) - 1
        x_ticks = np.round(grid_x[tick_idx, 0], 1);
        y_ticks = np.round(grid_y[0, tick_idx], 2);
        plt.imshow(np.argmax(action_probs, axis=2).T);
        plt.yticks(tick_idx, y_ticks);
        plt.xticks(tick_idx, x_ticks);
        plt.xlabel('Position');
        plt.ylabel('Velocity');
        plt.colorbar();
        plt.title("$\pi(s)$");
        plt.show()
