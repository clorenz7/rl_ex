from collections import namedtuple

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from cor_rl.factories import ffw_factory


InteractionResult = namedtuple(
    'InteractionResult',
    ['rewards', 'values', 'log_probs', 'entropies'],
)


class PolicyValueNetwork(nn.Module):

    def __init__(self, n_state, n_actions, hidden_layers):
        super().__init__()
        layer_sizes = [n_state] + hidden_layers
        self.base_layer = ffw_factory(layer_sizes, 'relu')

        n_hidden = hidden_layers[-1]

        self.policy_head = nn.Linear(n_hidden, n_actions)
        self.value_head = nn.Linear(n_hidden, 1)


    def forward(self, x):
        x = self.base_layer(x)

        logits = self.policy_head(x)

        action_probs = F.softmax(logits, dim=-1)

        value_est = self.value_head(x)

        return action_probs, value_est


def calc_n_step_returns(rewards, last_value_est, gamma):

    n_steps = len(rewards)
    n_step_returns = [0] * n_steps
    last_return = last_value_est
    for step_idx in reversed(range(n_steps)):
        last_return = rewards[step_idx] + gamma * last_return
        n_step_returns[step_idx] = last_return

    return n_step_returns


def calc_total_loss(results, gamma, clip=1.0, entropy_weight=0.01, device="cpu"):

    n_step_returns = calc_n_step_returns(
        results.rewards, results.values[-1], gamma
    )
    n_step_returns = torch.tensor(n_step_returns).to(device)

    value_est = torch.hstack(results.values[:-1])

    if clip is None or clip <= 0:
        value_loss = F.mse_loss(value_est, n_step_returns)
    else:
        value_loss = F.smooth_l1_loss(value_est, n_step_returns, beta=clip)

    advantage = n_step_returns - value_est
    # TODO: Is the negative correct?
    policy_loss = -torch.hstack(results.log_probs) * advantage

    loss = value_loss.sum() + policy_loss.sum()


    if entropy_weight > 0:
        entropy_loss = torch.hstack(results.entropies) * entropy_weight
        loss = loss + entropy_loss.sum()

    return loss



class BaseAgent:

    def __init__(self, agent_params):
        self.agent_params = agent_params or {}
        self.gamma = self.agent_params.get('gamma', 0.9)
        self.epsilon = self.agent_params.get('epsilon', 1e-8)
        self.min_vals = self.agent_params.get('min_vals', None)
        self.max_vals = self.agent_params.get('max_vals', None)
        self.n_actions = agent_params.get('n_actions')
        self.n_state = agent_params.get('n_state')
        self.grad_clip = self.agent_params.get('grad_clip', 1.0) or 0.0
        self.entropy_weight = self.agent_params.get('entropy_weight', 0.01)
        if self.min_vals and self.max_vals:
            self.mu = (
                torch.tensor(self.max_vals) + torch.tensor(self.min_vals)
            ) / 2.0
            self.sigma = (
                torch.tensor(self.max_vals) - torch.tensor(self.min_vals)
            ) / 2.0
        else:
            self.mu = self.sigma = None

    def bound(self, state, eps=1e-4):
        for i in range(len(state)):
            state[i] = max(
                min(state[i], self.max_vals[i]-eps),
                self.min_vals[i] + eps
            )
        return state

    def normalize_state(self, state):
        features = torch.tensor(state)
        if self.mu:
            features = (features - self.mu)/self.sigma

        return features.to(self.device)


class AdvantageActorCriticAgent(BaseAgent):
    def __init__(self, agent_params={}, train_params={}, device="cpu"):
        super().__init__(agent_params)
        self.last_state = None
        self.last_action = None

        self.device = device

        self.hidden_sizes = agent_params.get('hidden_sizes', None)
        self.n_state = agent_params.get('n_state', 2)

        self.train_params = train_params
        self.optimizer_name = self.train_params.pop('optimizer', 'adam').lower()

        alpha = self.train_params.pop('alpha', None)
        if alpha is not None:
            self.train_params['lr'] = alpha

        self.reset()

    def select_action(self, state=None):
        if state is not None:
            features = self.state_to_features(state)

        policy, value_est = self.net(features)

        pdf = Categorical(policy)
        action = pdf.sample()
        log_prob = pdf.log_prob(action)
        entropy = pdf.entropy()

        return action.item(), value_est, entropy, log_prob

    # def initialize(self, state):
    #     # TODO: Fix this
    #     self.last_state = state
    #     self.last_action = self.select_action(features=self.last_features)
    #     value_est = self.critic(self.last_features)
    #     self.last_value_est = value_est
    #     return self.last_action

    def state_to_features(self, state):
        return self.normalize_state(state)

    # def step(self, reward, state, debug=False):
    #     # TODO: Fix this
    #     features = self.state_to_features(state)

    #     last_log_prob = self.last_log_prob
    #     prev_value_est = self.last_value_est

    #     with torch.no_grad():
    #         value_est = self.critic(features)
    #     total_return_est = reward + self.gamma * value_est

    #     # Semi-gradient update
    #     delta = total_return_est - prev_value_est.item()

    #     policy_loss = delta * -last_log_prob
    #     value_loss = F.smooth_l1_loss(total_return_est, prev_value_est)
    #     loss = policy_loss + value_loss

    #     if self.optimizer is None:
    #         self.actor_optimizer.zero_grad(set_to_none=True)
    #         self.critic_optimizer.zero_grad(set_to_none=True)
    #         value_loss.backward(retain_graph=True)
    #         policy_loss.backward()
    #         self.critic_optimizer.step()
    #         self.actor_optimizer.step()
    #     else:
    #         self.optimizer.zero_grad(set_to_none=True)
    #         loss.backward()
    #         self.optimizer.step()

    #     next_action = self.select_action(features=features)
    #     value_est = self.critic(features)

    #     self.last_state = state
    #     self.last_features = features
    #     self.last_action = next_action
    #     self.last_value_est = value_est

    #     return next_action

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
        # TODO: Move this to an optimizer factory
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(),
                **self.train_params
            )
        elif self.optimizer_name == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                self.net.parameters(),
                **self.train_params
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                **self.train_params
            )

    def reset(self):
        self.net = PolicyValueNetwork(
            self.n_state, self.n_actions, self.hidden_sizes
        )

        self.set_optimizer()

    def checkpoint(self, file_name):
        torch.save(self.net, file_name)

    def load(self, file_name: str):
        self.net = torch.load(file_name).to(self.device)


    def get_grads(self, results: InteractionResult):

        #---- Compute the loss
        loss = calc_total_loss(
            results, self.gamma, clip=self.grad_clip,
            entropy_weight=self.entropy_weight, device=self.device
        )
        loss.backward()

        # Extract the grads
        # from collections import OrderedDict
        # grads = OrderedDict()
        grads = {}
        for name, param in self.net.named_parameters():
            grads[name] = param.grad.detach()

        return grads

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update(self, state_dict):
        self.net.load_state_dict(state_dict)

    def get_parameters(self):
        return self.net.state_dict()

    def set_grads(self, grads):
        for name, param in self.net.named_parameters():
            param.grad = grads[name]

    def backward(self):
        self.optimizer.step()
        # TODO: Add learning rate scheduler


def interact(env, agent, t_max=5, state=None, output_gif=False):
    """
    Does t_max steps of the agent in the environment.
    This is single threaded.
    """

    frame_buffer = []
    results = InteractionResult([], [], [], [])
    terminated = False
    t = 0

    if state is None:
        state, info = env.reset()
        action_idx, value_est, entropy, log_prob = agent.select_action(state)
        if output_gif:
            frame_buffer.append(env.render())
        state, reward, terminated, _, _ = env.step(action_idx)
        results.rewards.append(reward)
        results.values.append(value_est)
        results.log_probs.append(log_prob)
        results.entropies.append(entropy)
        if output_gif:
            frame_buffer.append(env.render())

        t += 1

    while t < t_max and not terminated:
        # TODO: I think I need to decouple reward and state...
        # Think through input and outputs of networks and steps
        # action_idx = agent.step(reward, state)

        # Run network
        action_idx, value_est, entropy, log_prob = agent.select_action(state)
        state, reward, terminated, _, _ = env.step(action_idx)

        results.rewards.append(reward)
        results.values.append(value_est)
        results.log_probs.append(log_prob)
        results.entropies.append(entropy)

        if output_gif:
            frame_buffer.append(env.render())
        t += 1

    # if terminated:
    #     agent.finish(reward)
    #     state = None

    # TODO: I think I need a no_grad estimate of state value...
    if terminated:
        value_est = 0.0
        state = None
    else:
        with torch.no_grad():
            action_idx, value_est, _, _ = agent.select_action(state)
        value_est = value_est.item()

    results.values.append(value_est)

    return results, state, terminated, frame_buffer


def agent_env_task(agent, env, parameters, state, t_max=5):

    if parameters is not None:
        agent.update(parameters)

    agent.zero_grad()

    results, state, terminated, _ = interact(
        env, agent, t_max=t_max, state=state
    )

    # This will run back prop
    grads = agent.get_grads(results)

    # TODO: Add total reward!

    return {
        'grads': grads,
        'state': state,
        'total_reward': sum(results.rewards),
        'terminated': terminated,
        'n_steps': len(results.rewards),
    }


def train_loop(global_agent: AdvantageActorCriticAgent, agents, envs, step_limit=10000, episode_limit=None,
               log_interval=1e9, solved_thresh=None, max_ep_steps=10000):

    solved_thresh = solved_thresh or float('inf')
    total_steps = 0
    n_episodes = 0
    ep_steps, ep_reward = 0, 0
    avg_reward = 0
    n_threads = len(agents)
    states = [None] * n_threads

    while total_steps < step_limit and n_episodes < episode_limit:
        params = global_agent.get_parameters()
        for t_idx in range(n_threads):
            agent = agents[t_idx]
            task_result = agent_env_task(
                agent, envs[t_idx], params, states[t_idx], t_max=500
            )
            n_steps = task_result['n_steps']
            ep_reward += task_result['total_reward']
            ep_steps += n_steps
            total_steps += n_steps
            if ep_steps > max_ep_steps:
                # Terminate early
                states[t_idx] = None
            else:
                states[t_idx] = task_result['state']
            if states[t_idx] is None:
                avg_reward = 0.99 * avg_reward + 0.01 * ep_reward
                n_episodes += 1
                # print(f"Episode {total_episodes} Terminated after {ep_steps} steps. Total steps: {total_steps}")
                if (n_episodes % log_interval) == 0:
                    print(
                        f'Episode {n_episodes}\tLast reward: {ep_reward:.2f}\t'
                        f'Average reward: {avg_reward:.2f}'
                    )
                ep_steps = ep_reward = 0

            global_agent.set_grads(task_result['grads'])
            global_agent.backward()

        if avg_reward > solved_thresh:
            print(f'Episode {n_episodes}\tLast reward: {ep_reward:.2f}\tAverage reward: {avg_reward:.2f}')
            print("PROBLEM SOLVED!")
            break





