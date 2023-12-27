from collections import namedtuple
from contextlib import contextmanager
import multiprocessing
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from cor_rl.factories import (
    ffw_factory,
    optimizer_factory,
)

from cor_rl import environments
from cor_rl.atari import PolicyValueImageNetwork
from cor_rl.agents.a2c import (
    InteractionResult,
    AdvantageActorCriticAgent
)


# InteractionResult = namedtuple(
#     'InteractionResult',
#     ['rewards', 'values', 'log_probs', 'entropies'],
# )

# EPS = np.finfo(np.float32).eps.item()


# class PolicyValueNetwork(nn.Module):

#     def __init__(self, n_state, n_actions, hidden_layers):
#         super().__init__()
#         layer_sizes = [n_state] + hidden_layers
#         self.base_layer = ffw_factory(layer_sizes, 'relu', 'relu')

#         n_hidden = hidden_layers[-1]

#         self.policy_head = nn.Linear(n_hidden, n_actions)
#         self.value_head = nn.Linear(n_hidden, 1)

#     def forward(self, x):
#         x = self.base_layer(x)

#         logits = self.policy_head(x)
#         action_probs = F.softmax(logits, dim=-1)

#         value_est = self.value_head(x)

#         return action_probs, value_est


# def calc_n_step_returns(rewards, last_value_est, gamma, reward_clip=None):

#     n_steps = len(rewards)
#     n_step_returns = [0] * n_steps
#     last_return = last_value_est
#     for step_idx in reversed(range(n_steps)):
#         reward = rewards[step_idx]
#         # Mnih paper clipped rewards to +-1 to account for different game scales
#         if reward_clip is not None:
#             reward = max(min(reward, reward_clip), -reward_clip)
#         last_return = reward + gamma * last_return
#         n_step_returns[step_idx] = last_return

#     return n_step_returns


# class BaseAgent:

#     def __init__(self, agent_params):
#         self.agent_params = agent_params or {}
#         self.gamma = self.agent_params.get('gamma', 0.9)
#         self.epsilon = self.agent_params.get('epsilon', 1e-8)
#         self.min_vals = self.agent_params.get('min_vals', None)
#         self.max_vals = self.agent_params.get('max_vals', None)
#         self.n_actions = agent_params.get('n_actions')
#         self.n_state = agent_params.get('n_state')
#         self.grad_clip = self.agent_params.get('grad_clip', 1.0) or 0.0
#         self.reward_clip = self.agent_params.get('reward_clip')
#         self.entropy_weight = self.agent_params.get('entropy_weight', 0.01)
#         if self.min_vals and self.max_vals:
#             self.mu = (
#                 torch.tensor(self.max_vals) + torch.tensor(self.min_vals)
#             ) / 2.0
#             self.sigma = (
#                 torch.tensor(self.max_vals) - torch.tensor(self.min_vals)
#             ) / 2.0
#         else:
#             self.mu = self.sigma = None

#     def bound(self, state, eps=1e-4):
#         for i in range(len(state)):
#             state[i] = max(
#                 min(state[i], self.max_vals[i]-eps),
#                 self.min_vals[i] + eps
#             )
#         return state

#     def normalize_state(self, state):
#         features = torch.from_numpy(state).float()
#         if self.mu:
#             features = (features - self.mu)/self.sigma

#         return features.to(self.device)


# class AdvantageActorCriticAgent(BaseAgent):
#     def __init__(self, agent_params={}, train_params={}, device="cpu"):
#         super().__init__(agent_params)
#         self.device = device

#         self.hidden_sizes = agent_params.get('hidden_sizes', None)
#         self.input_type = agent_params.get('input_type', 'vector').lower()

#         self.train_params = train_params
#         self.optimizer_name = self.train_params.pop('optimizer', 'adam').lower()

#         self.norm_returns = self.agent_params.get('norm_returns', False)

#         self.reset()

#     def select_action(self, state=None):
#         if state is not None:
#             features = self.state_to_features(state)

#         policy, value_est = self.net(features)

#         pdf = Categorical(policy)
#         action = pdf.sample()
#         log_prob = pdf.log_prob(action)
#         entropy = pdf.entropy()

#         return action.item(), value_est, entropy, log_prob

#     def state_to_features(self, state):
#         return self.normalize_state(state)

#     def reset(self):
#         if self.input_type == 'vector':
#             self.net = PolicyValueNetwork(
#                 self.n_state, self.n_actions, self.hidden_sizes
#             )
#         elif self.input_type == 'image':
#             self.net = PolicyValueImageNetwork(self.n_actions)
#         else:
#             raise ValueError(f"{self.input_type} not a valid input type!")

#         self.optimizer = optimizer_factory.get(
#             self.optimizer_name, self.train_params, self.net
#         )

#     def checkpoint(self, file_name):
#         torch.save(self.net, file_name)

#     def load(self, file_name: str):
#         self.net = torch.load(file_name).to(self.device)

#     def calculate_loss(self, results):
#         n_step_returns = calc_n_step_returns(
#             results.rewards, results.values[-1], self.gamma, self.reward_clip
#         )
#         n_step_returns = torch.tensor(n_step_returns).to(self.device)

#         value_est = torch.hstack(results.values[:-1])

#         if self.norm_returns:
#             # Pytorch reference implementation does this, dunno exactly why
#             # But my intuition is that it helps deal with the way
#             # total return increases as algo improves
#             std = n_step_returns.std() + EPS
#             n_step_returns = (n_step_returns - n_step_returns.mean()) / std

#         if self.grad_clip is None or self.grad_clip <= 0:
#             value_loss = F.mse_loss(value_est, n_step_returns)
#         else:
#             value_loss = F.smooth_l1_loss(
#                 value_est, n_step_returns,
#                 beta=self.grad_clip, reduction="none"
#             )

#         # Advantage is a semi-gradient update
#         advantage = n_step_returns - value_est.detach()
#         policy_loss = -torch.hstack(results.log_probs) * advantage

#         loss = value_loss.sum() + policy_loss.sum()

#         if self.entropy_weight > 0:
#             entropy_loss = torch.hstack(results.entropies)
#             loss = loss + entropy_loss.sum() * self.entropy_weight

#         return loss

#     def set_parameters(self, state_dict):
#         tensor_state = {}
#         for key, val in state_dict.items():
#             tensor_state[key] = torch.tensor(val)
#         self.net.load_state_dict(tensor_state)

#     def get_parameters(self):
#         state_dict = self.net.state_dict()
#         for key, val in state_dict.items():
#             state_dict[key] = val.tolist()

#         return state_dict

#     def set_grads(self, grads):
#         for name, param in self.net.named_parameters():
#             param.grad = grads[name]

#     def get_grads(self, results: InteractionResult):
#         # Compute the loss
#         loss = self.calculate_loss(results)
#         loss.backward()

#         grads = {}
#         for name, param in self.net.named_parameters():
#             grads[name] = param.grad.detach()

#         return grads

#     def backward(self):
#         self.optimizer.step()
#         # TODO: Add learning rate scheduler

#         self.optimizer.zero_grad()

#     def zero_grad(self):
#         self.optimizer.zero_grad()


def interact(env, agent, t_max=5, state=None, output_frames=False):
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
        if output_frames:
            frame_buffer.append(env.render())

    while t < t_max and not terminated:
        # Run network
        action_idx, value_est, entropy, log_prob = agent.select_action(state)
        state, reward, terminated, _, _ = env.step(action_idx)

        results.rewards.append(reward)
        results.values.append(value_est)
        results.log_probs.append(log_prob)
        results.entropies.append(entropy)

        if output_frames:
            frame_buffer.append(env.render())
        t += 1

    if terminated:
        value_est = 0.0
        state = None
    else:
        # Get an estimate of the value of the final state
        with torch.no_grad():
            action_idx, value_est, _, _ = agent.select_action(state)
        value_est = value_est.item()

    results.values.append(value_est)

    return results, state, terminated, frame_buffer


def agent_env_task(agent, env, parameters, state, t_max=5,
                   output_frames=False):

    if parameters is not None:
        agent.set_parameters(parameters)

    agent.zero_grad()

    results, state, terminated, frames = interact(
        env, agent, t_max=t_max, state=state, output_frames=output_frames
    )

    # This will run back prop
    grads = agent.get_grads(results)

    output = {
        'grads': grads,
        'state': state,
        'total_reward': sum(results.rewards),
        'terminated': terminated,
        'n_steps': len(results.rewards),
    }

    if output_frames:
        output['frames'] = frames

    return output


def train_loop(global_agent: AdvantageActorCriticAgent, agents, envs, step_limit=10000, episode_limit=None,
               log_interval=1e9, solved_thresh=None, max_ep_steps=10000, t_max=10000, debug=False,
               avg_decay=0.95):
    """
    This is a single threaded (serial) training loop with multiple agents
    """
    start_time = time.time()

    solved_thresh = solved_thresh or float('inf')
    total_steps = 0
    n_episodes = 0
    ep_steps, ep_reward = 0, 0
    avg_reward = 0
    solved = False

    n_threads = len(agents)
    states = [None] * n_threads
    print("")

    while total_steps < step_limit and n_episodes < episode_limit:
        params = global_agent.get_parameters()
        if debug:
            for key, val in params.items():
                print(f'{key}: {torch.tensor(val).flatten()[:2]}')
            w_idx = 0
            w_params = agents[w_idx].get_parameters()
            state = states[w_idx]
            print([] if state is None else state.tolist())
            for key, val in w_params.items():
                print(f'{key}: {torch.tensor(val).flatten()[:2]}')

        for t_idx in range(n_threads):
            agent = agents[t_idx]
            task_result = agent_env_task(
                agent, envs[t_idx], params, states[t_idx], t_max=t_max
            )
            n_steps = task_result['n_steps']
            ep_reward += task_result['total_reward']
            ep_steps += n_steps
            total_steps += n_steps
            if ep_steps >= max_ep_steps:
                # Terminate early
                states[t_idx] = None
            else:
                states[t_idx] = task_result['state']

            if states[t_idx] is None:
                last_reward = ep_reward
                avg_reward = (
                    avg_decay * avg_reward + (1.0 - avg_decay) * ep_reward
                )
                n_episodes += 1
                if (n_episodes % log_interval) == 0:
                    print(
                        f'Episode {n_episodes}\tLast reward: {last_reward:.2f}\t'
                        f'Average reward: {avg_reward:.2f}'
                    )
                ep_steps = ep_reward = 0

            if debug:
                print(f"State: {states[t_idx]}")
                print("\nWorker Agent Grads:")
                for key, val in task_result['grads'].items():
                    print(f'{key}: {torch.tensor(val).flatten()[:2]}')

            global_agent.set_grads(task_result['grads'])
            global_agent.backward()

        if avg_reward > solved_thresh:
            print(f'Episode {n_episodes}\tLast reward: {last_reward:.2f}\tAverage reward: {avg_reward:.2f}')
            print(f"PROBLEM SOLVED in {time.time() - start_time:0.1f}sec")
            solved = True
            break

    if not solved:
        print(f"Finished in {time.time() - start_time:0.1f}sec")

    return global_agent, solved


@contextmanager
def piped_workers(n_workers, worker_func, worker_args):
    """
    Context manager to manage a set of training worker threads.

    Pipes and processes are automatically closed and terminated.
    """

    parent_conns, child_conns = [], []
    worker_processes = []

    for i in range(n_workers):
        parent_conn, child_conn = multiprocessing.Pipe()
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

        worker_process = multiprocessing.Process(
            target=worker_func,
            args=[i, child_conn, *worker_args]
        )
        worker_processes.append(worker_process)
        worker_process.start()

    try:
        yield parent_conns
    finally:
        for i in range(n_workers):
            worker_processes[i].terminate()
            parent_conns[i].close()
            child_conns[i].close()


def worker_thread(task_id, conn, agent_params, train_params, env_params):
    """
    This is the thread which trains an agent based on messages from the parent.
    It receives the current global model parameters and
    returns gradient updates to apply to the global model.
    """

    env_name = env_params['env_name']
    seed = env_params.get('seed', 8888)
    max_steps_per_episode = env_params.get('max_steps_per_episode', 1e9)
    env = environments.factory.get(env_name)
    ep_steps = 0
    if seed:
        task_seed = seed + task_id * 10
        env.reset(seed=task_seed)
        torch.manual_seed(task_seed)
    state = None  # Force reset to match previous work

    agent = AdvantageActorCriticAgent(agent_params, train_params)
    torch.manual_seed(task_seed)  # Reset seed to match previous work

    while True:
        # Receive a task from the parent process
        task = conn.recv()
        task_type = task.get('type', '')

        if task_type == 'train':
            # Allow controller to reset the environment
            if task.get('reset', False):
                state = None

            # Train the agent for a few steps
            result = agent_env_task(
                agent, env, task['params'], state,
                t_max=task['max_steps']
            )
            # Update the state for next time
            state = result['state']
            # Keep track of steps for this episode and end if too many
            ep_steps += result['n_steps']
            if ep_steps >= max_steps_per_episode:
                state = None
                result['terminated'] = True

            # Restart counter if terminated
            if result['terminated']:
                ep_steps = 0

            # Send result to parent process
            conn.send(result)

        elif task_type == 'params':
            params = agent.get_parameters()
            params['seed'] = [task_seed, seed]
            params['state'] = [] if state is None else state.tolist()
            conn.send(params)
        elif task_type == 'state':
            conn.send([] if state is None else state.tolist())
        elif task_type == 'STOP':
            conn.send("FINISH HIM!")
            break


def _show_params(params, msg_pipe):
    print("Global Agent Params:")
    for key, val in params.items():
        print(f'{key}: {torch.tensor(val).flatten()[:2]}')
    msg_pipe.send({'type': 'params'})
    w_params = msg_pipe.recv()
    for key, val in w_params.items():
        print(f'{key}: {torch.tensor(val).flatten()[:2]}')


def _show_grads(result):
    print(f"State: {result['state']}")
    print("\nWorker Agent Grads:")
    for key, val in result['grads'].items():
        print(f'{key}: {torch.tensor(val).flatten()[:2]}')
    import ipdb; ipdb.set_trace()


def train_loop_parallel(n_workers, agent_params, train_params, env_name,
                        steps_per_batch=5, total_step_limit=10000,
                        episode_limit=None, max_steps_per_episode=10000,
                        solved_thresh=None, log_interval=1e9, seed=8888,
                        avg_decay=0.95, debug=False):
    """
    Training loop which sets up multiple worker threads which compute
    gradients in parallel.
    """
    start_time = time.time()

    solved_thresh = solved_thresh or float('inf')
    total_steps = total_episodes = 0
    ep_steps, ep_reward = [0]*n_workers, [0]*n_workers
    avg_reward = 0
    solved = False
    episode_limit = episode_limit or 1e9
    keep_training = True

    env_params = {
        'env_name': env_name,
        'seed': seed,
        'max_steps_per_episode': max_steps_per_episode,
    }

    # Seed and create the global agent
    if seed:
        torch.manual_seed(seed)
    global_agent = AdvantageActorCriticAgent(agent_params, train_params)

    worker_args = [agent_params, train_params, env_params]
    with piped_workers(n_workers, worker_thread, worker_args) as msg_pipes:
        while keep_training:
            params = global_agent.get_parameters()
            if debug:
                _show_params(params, msg_pipes[0])

            # Signal each of the workers to generate a batch of data
            payload = {
                'type': 'train',
                'max_steps': steps_per_batch,
                'params': params
            }
            for w_idx in range(n_workers):
                msg_pipes[w_idx].send(payload)

            # Get the result from each worker and update the model
            for w_idx in range(n_workers):
                result = msg_pipes[w_idx].recv()
                if debug:
                    _show_grads(result)

                global_agent.set_grads(result['grads'])
                global_agent.backward()

                # Update counters and print out if necessary
                total_steps += result['n_steps']
                ep_reward[w_idx] += result['total_reward']

                if result['terminated']:
                    total_episodes += 1
                    last_reward = ep_reward[w_idx]
                    avg_reward = (
                        avg_decay * avg_reward +
                        (1.0 - avg_decay) * last_reward
                    )
                    solved = avg_reward > solved_thresh
                    ep_steps[w_idx] = ep_reward[w_idx] = 0
                    if (total_episodes % log_interval) == 0:
                        print(
                            f'Episode {total_episodes}\t'
                            f'Last reward: {last_reward:.2f}\t'
                            f'Average reward: {avg_reward:.2f}'
                        )
                if solved:
                    break

            keep_training = (
                not solved and
                total_steps < total_step_limit and
                total_episodes < episode_limit
            )

    if solved:
        print(
            f'Episode {total_episodes}\tLast reward: {last_reward:.2f}\t'
            f'Average reward: {avg_reward:.2f}'
        )
        print(f"PROBLEM SOLVED in {time.time() - start_time:0.1f} sec!")
    else:
        print(f"Aborted after {time.time() - start_time:0.1f}sec")

    return global_agent, solved
