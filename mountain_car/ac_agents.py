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


class MountainCarActorCriticAgent:

    def __init__(self, n_actions, agent_params={}, train_params={}, device="cpu"):
        self.last_state = None
        self.last_action = None

        self.device = device
        self.gamma = agent_params.get('gamma', 0.9)
        self.n_actions = n_actions
        self.min_vals = MIN_VALS
        self.max_vals = MAX_VALS
        self.n_hidden = agent_params.get('n_hidden', 512)
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
        self.optimizer.zero_grad(set_to_none=True)
        if debug:
            weights = self.net.base_layer[0].weight.clone().detach()

        features = self.state_to_features(state)

        last_log_prob = self.last_log_prob
        prev_value_est = self.last_value_est

        # with torch.no_grad():
        #     next_action, value_est = self.get_action_and_value(features=features)
        with torch.no_grad():
            value_est = self.critic(features)

        total_return_est = reward + self.gamma * value_est.item()

        delta = total_return_est - prev_value_est

        policy_loss = delta * -last_log_prob
        value_loss = F.smooth_l1_loss(torch.tensor([total_return_est], device=self.device), prev_value_est)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad(set_to_none=True)
        # if debug:
        #     loss.retain_grad()
        loss.backward()
        self.optimizer.step()

        # import ipdb; ipdb.set_trace()
        # next_action, value_est = self.get_action_and_value(features=features)
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
        self.optimizer.zero_grad(set_to_none=True)
        last_log_prob = self.last_log_prob
        prev_value_est = self.last_value_est
        delta = reward - prev_value_est

        policy_loss = delta * -last_log_prob
        value_loss = F.smooth_l1_loss(
            torch.tensor([reward], device=self.device), prev_value_est
        )
        loss = policy_loss + value_loss
        loss.backward()
        self.optimizer.step()

    def reset(self):
        # self.net = Policy(
        #     self.n_actions, self.n_hidden, self.n_state
        # ).to(self.device)
        # params = self.net.parameters()

        self.actor = nn.Sequential(
            nn.Linear(self.n_state, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_actions),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(self.n_state, self.n_hidden),
            nn.ReLU(),
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