from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from .base import BaseAgent
from cor_rl.factories import ffw_factory
from cor_rl.factories import optimizer_factory


InteractionResult = namedtuple(
    'InteractionResult',
    ['rewards', 'values', 'log_probs', 'entropies'],
)

EPS = np.finfo(np.float32).eps.item()


def calc_n_step_returns(rewards, last_value_est, gamma, reward_clip=1e9):

    n_steps = len(rewards)
    n_step_returns = [0] * n_steps
    last_return = last_value_est
    for step_idx in reversed(range(n_steps)):
        reward = rewards[step_idx]
        # Mnih paper clipped rewards to +-1 to account for different game scales
        reward = max(min(reward, reward_clip), -reward_clip)
        last_return = reward + gamma * last_return
        n_step_returns[step_idx] = last_return

    return n_step_returns


class PolicyValueNetwork(nn.Module):

    def __init__(self, n_state, n_actions, hidden_layers):
        super().__init__()
        layer_sizes = [n_state] + hidden_layers
        self.base_layer = ffw_factory(layer_sizes, 'relu', 'relu')

        n_hidden = hidden_layers[-1]

        self.policy_head = nn.Linear(n_hidden, n_actions)
        self.value_head = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.base_layer(x)

        logits = self.policy_head(x)
        action_probs = F.softmax(logits, dim=-1)

        value_est = self.value_head(x)

        return action_probs, value_est


class AdvantageActorCriticAgent(BaseAgent):
    def __init__(self, agent_params={}, train_params={}, device="cpu"):
        super().__init__(agent_params)
        self.device = device

        self.hidden_sizes = agent_params.get('hidden_sizes', None)

        self.train_params = dict(train_params)
        self.optimizer_name = self.train_params.pop('optimizer', 'adam').lower()
        if agent_params.get("shared", False):
            self.optimizer_name += "_shared"

        self.norm_returns = self.agent_params.get('norm_returns', False)
        self.clip_grad_norm = self.agent_params.get('clip_grad_norm', 0.0)
        self.n_exp_steps = self.agent_params.get('n_exp_steps', None)
        self.max_entropy = np.log2(self.n_actions)
        self.value_loss_factor = self.agent_params.get('value_loss_factor', 1.0)

        self.reset()

    def params(self):
        return {
            'clip_grad_norm': self.clip_grad_norm,
            'reward_clip': self.reward_clip,
            'lr': self.optimizer.param_groups[0]['lr'],
            'optimizer': self.optimizer_name,
            'value_loss_factor': self.value_loss_factor,
        }

    def select_action(self, state=None, lock=None):
        if state is not None:
            features = self.state_to_features(state)

        if lock is None:
            policy, value_est = self.net(features)
        else:
            with lock:
                policy, value_est = self.net(features)

        pdf = Categorical(policy)
        action = pdf.sample()
        log_prob = pdf.log_prob(action)
        entropy = pdf.entropy()

        return action.item(), value_est, entropy, log_prob

    def construct_net(self):
        return PolicyValueNetwork(
            self.n_state, self.n_actions, self.hidden_sizes
        )

    def reset(self):
        self.net = self.construct_net()

        self.optimizer = optimizer_factory.get(
            self.optimizer_name, self.train_params, self.net
        )

    def checkpoint(self, file_name):
        torch.save(self.net, file_name)

    def load(self, file_name: str):
        self.net = torch.load(file_name).to(self.device)

    def calculate_loss(self, results):
        n_step_returns = calc_n_step_returns(
            results.rewards, results.values[-1], self.gamma, self.reward_clip
        )
        n_steps = len(n_step_returns)
        n_step_returns = torch.tensor(n_step_returns).to(self.device)

        value_est = torch.hstack(results.values[:-1])

        if self.norm_returns:
            # Pytorch reference implementation does this, dunno exactly why
            # But my intuition is that it helps deal with the way
            # total return increases as algo improves
            std = n_step_returns.std() + EPS
            n_step_returns = (n_step_returns - n_step_returns.mean()) / std

        if self.value_loss_clip is None or self.value_loss_clip <= 0:
            value_loss = F.mse_loss(value_est, n_step_returns, reduction="none")
        else:
            value_loss = F.smooth_l1_loss(
                value_est, n_step_returns,
                beta=self.value_loss_clip, reduction="none"
            )

        # Advantage is a semi-gradient update
        advantage = n_step_returns - value_est.detach()
        policy_loss = -torch.hstack(results.log_probs) * advantage

        loss = self.value_loss_factor * value_loss.sum() + policy_loss.sum()

        if self.n_exp_steps:
            loss = loss * (self.n_exp_steps / n_steps)

        if self.entropy_weight > 0:
            entropy_loss = torch.hstack(results.entropies)
            loss = loss - entropy_loss.sum() * self.entropy_weight

        return loss

    def set_parameters(self, state_dict, copy=False):
        if copy:
            tensor_state = {}
            for key, val in state_dict.items():
                tensor_state[key] = torch.tensor(val)

            self.net.load_state_dict(tensor_state)
        else:
            self.net.load_state_dict(state_dict)

    def get_parameters(self, tolist=False):
        if tolist:
            state_dict = self.net.state_dict()
            for key, val in state_dict.items():
                state_dict[key] = val.tolist()
        else:
            return self.net.state_dict()

        return state_dict

    def sync_grads(self, other_net):

        for self_p, other_p in zip(self.net.parameters(), other_net.parameters()):
            if other_p.grad is None:
                other_p._grad = self_p.grad

    def calc_loss_and_backprop(self, results: InteractionResult):
        # Compute the loss
        loss = self.calculate_loss(results)
        loss.backward()

        # if self.clip_grad_norm > 0:
        norm_val = nn.utils.clip_grad_norm_(
            self.net.parameters(), self.clip_grad_norm or 1e9
        )
        return loss, norm_val


    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
