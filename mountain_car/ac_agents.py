from functools import partial

import torch
from torch import nn


from torch.distributions import Categorical
import torch.nn.functional as F

MIN_VALS = [-1.2, -0.07]
MAX_VALS = [0.6, 0.07]


class Policy(nn.Module):

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


def net_from_layer_sizes(layer_sizes, activation=nn.ELU, final_activation=None):
    layers = []
    for ii in range(len(layer_sizes)-1):
        layers.append(nn.Linear(layer_sizes[ii], layer_sizes[ii+1]))
        layers.append(activation())

    # Remove final activation and re-add it
    layers.pop()
    if final_activation is not None:
        layers.append(final_activation())

    return nn.Sequential(*layers)


class MountainCarActorCriticAgent:

    def __init__(self, n_actions, agent_params={}, train_params={}, device="cpu"):
        self.last_state = None
        self.last_action = None

        self.single_grad = agent_params.get("single_grad", True)

        self.device = device
        self.gamma = agent_params.get('gamma', 0.9)
        self.n_actions = n_actions
        self.min_vals = MIN_VALS
        self.max_vals = MAX_VALS
        self.n_hidden = agent_params.get('n_hidden', None)

        self.actor_layers = agent_params.get('actor_layers', [48, 48])
        self.critic_layers = agent_params.get('critic_layers', [384, 384])

        self.n_state = agent_params.get('n_state', 2)
        self.mu = (torch.tensor(self.max_vals) + torch.tensor(self.min_vals))/2.0
        self.sigma = (torch.tensor(self.max_vals) - torch.tensor(self.min_vals))/2.0
        self.train_params = train_params
        self.optimizer_name = self.train_params.pop('optimizer', 'adam').lower()

        alpha = self.train_params.pop('alpha', None)
        if alpha is not None:
            self.train_params['lr'] = alpha

        self.reset()

    def bound(self, state, eps=1e-4):
        for i in range(len(state)):
            state[i] = max(
                min(state[i], self.max_vals[i]-eps),
                self.min_vals[i] + eps
            )
        return state

    # def state_to_features(self, state):
    #     return torch.from_numpy(state).float().to(self.device)

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
        # self.last_action, value_est = self.get_action_and_value(features=self.last_features)
        self.last_action = self.select_action(features=self.last_features)
        value_est = self.critic(self.last_features)
        self.last_value_est = value_est
        return self.last_action

    def state_to_features(self, state):
        features = (
            (torch.tensor(state) - self.mu)/self.sigma
        ).to(self.device)

        return features

    def step(self, reward, state, debug=False):
        if debug:
            weights = self.net.base_layer[0].weight.clone().detach()

        features = self.state_to_features(state)

        last_log_prob = self.last_log_prob
        prev_value_est = self.last_value_est

        if self.single_grad:
            with torch.no_grad():
                value_est = self.critic(features)
        else:
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
        # if debug:
        #     loss.retain_grad()

        next_action = self.select_action(features=features)
        value_est = self.critic(features)

        # if debug:
        #     # This is sanity checking.
        #     grad = self.last_features
        #     actual_update = self.net[0].weight.detach() - weights
        #     exp_update = 2 * self.optimizer.param_groups[0]['lr'] * delta.clone().detach() * grad

        #     with torch.no_grad():
        #         updated_value = self.net(self.last_features)[self.last_action]

        #     value_delta = updated_value - last_value
        #     weight_delta = torch.abs(actual_update[self.last_action, :] - exp_update).max()

        self.last_state = state
        self.last_features = features
        self.last_action = next_action
        self.last_value_est = value_est

        return self.last_action

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
            actor_layer_sizes = [self.n_state] + self.actor_layers + [self.n_actions]
            self.actor = net_from_layer_sizes(
                actor_layer_sizes,
                final_activation=partial(nn.Softmax, dim=-1)
            ).to(self.device)
            critic_layer_sizes = [self.n_state] + self.critic_layers + [1]
            self.critic = net_from_layer_sizes(
                critic_layer_sizes,
            ).to(self.device)
            self.optimizer = None
            self.actor_optimizer = torch.optim.AdamW(
                self.actor.parameters(),
                **self.train_params['actor']
            )
            self.critic_optimizer = torch.optim.AdamW(
                self.critic.parameters(),
                **self.train_params.get('critic', {})
            )


