"""
Mountain car
- Initial version is tile coding with linear function approx.
"""
import argparse
from copy import deepcopy
import datetime
import json
import pprint
import time

import gymnasium as gym
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter;
import numpy as np
import torch

import q_agents
import ac_agents


def experiment_loop(env, agent, seed=101, n_runs=100, n_episodes=500,
                    eval_episodes={}, max_steps=[10000, 10000], verbose=True,
                    render=False):

    run_steps = np.zeros((n_episodes, n_runs))

    if isinstance(max_steps, int):
        max_steps = [max_steps]*2

    step_limit = np.linspace(max_steps[0], max_steps[1], n_episodes)

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
                if render:
                    if (s_idx+1) % 500 == 0:
                        print(f"Episode {e_idx} Step {s_idx+1}")
                trajectory.append(state)
                s_idx += 1
                actions.append(action_idx)

                # Pause if loop might be infinite
                if s_idx >= step_limit[e_idx]:
                    final_state = "aborted"
                    elap = round((time.time() - st)/60, 2)
                    # print(f"Break at {s_idx} after {elap}min")
                    # agent.evaluate_q()
                    # np_traj = np.vstack(trajectory).T
                    # import ipdb; ipdb.set_trace()
                    break
            if terminated:
                final_state = "reached goal"
                agent.finish(reward)
            elap = round((time.time() - st)/60, 2)
            # agent.evaluate_q()
            # Record result and display if desired
            run_steps[e_idx, r_idx] = s_idx
            if verbose or (e_idx + 1) % 100 == 0:
                print(f"Run {r_idx+1} Episode {e_idx+1} {final_state} at step {s_idx} in {elap}min")
            if e_idx in eval_episodes:
                q_vals = agent.evaluate_q()

            state, info = env.reset()
        agent.reset()

    return run_steps


class ExperimentParams:

    DEFAULT_SIM = {
        'n_episodes': 80,
        "n_runs": 8
    }

    def __init__(self, agent_params={}, train_params={},
                 simulation_params=None):
        self.agent_params = agent_params
        self.train_params = train_params
        self.simulation_params = simulation_params or dict(self.DEFAULT_SIM)

    def to_dict(self):
        return dict(
            agent_params=self.agent_params,
            train_params=self.train_params,
            simulation_params=self.simulation_params,
        )

    def copy_and_update_params(self, keys, vals):
        params = dict(
            agent_params=deepcopy(self.agent_params),
            train_params=deepcopy(self.train_params)
        )

        for key, vals in zip(keys, vals):
            start, param = key.split(".", 1)
            params[start][param] = vals

        return params['agent_params'], params['train_params']


class AgentFactory:
    _AGENTS = {
        'deepq': q_agents.FFWQAgent,
        'qtiles': q_agents.TiledLinearQAgent,
        'actor-critic': ac_agents.MountainCarActorCriticAgent,
    }

    def __init__(self, agent_type, n_actions, device="cpu"):
        self.agent_type = agent_type.lower()
        self.n_actions = n_actions
        self.device = device

    def get(self, agent_params, train_params):
        constructor = self._AGENTS[self.agent_type]
        return constructor(
            self.n_actions, agent_params, train_params, device=self.device
        )


def main():

    parser = argparse.ArgumentParser(
        description="Train an RL Agent for Mountain Car"
    )
    parser.add_argument(
        '-j', "--json", type=str, default="",
        help="Json experiment parameters"
    )
    parser.add_argument(
        '-r', "--render", action="store_true",
        help="Render the agent in the environment"
    )
    parser.add_argument(
        '-g', "--no_gpu", action="store_true",
        help="Turn off GPU acceleration"
    )

    cli_args = parser.parse_args()
    date_time = datetime.datetime.now().strftime('%Y_%b_%d_%H_%M')

    if cli_args.json:
        with open(cli_args.json, 'r') as fp:
            json_params = json.load(fp)
    else:
        json_params = {}
    default_agent_type = "actor-critic" if "ac" in cli_args.json else "deepq"
    pprint.pprint(json_params)

    agent_type = json_params.pop("agent_type", None)
    if agent_type is None:
        print(f"WARNING! Agent Type no specified, using {default_agent_type}")
        agent_type = default_agent_type

    param_study = json_params.pop('param_study', {})
    exp_params = ExperimentParams(**json_params)

    render_mode = "human" if cli_args.render else None
    env = gym.make('MountainCar-v0', render_mode=render_mode)
    n_actions = env.action_space.n

    device = torch.device("cpu")
    if not cli_args.no_gpu:
        if torch.backends.mps.is_available():
            print("Accelerating with MPS!")
            device = torch.device("mps")
        elif torch.cuda.is_available():
            print("Accelerating with CUDA!")
            device = torch.device("cuda")

    factory = AgentFactory(agent_type, n_actions, device=device)

    if param_study:
        param_keys = [k for k in param_study.keys()]
        n_vals = [len(v) for v in param_study.values()]
        # Limit to first two dimensions for now
        param_keys = param_keys[:2]
        n_vals = n_vals[:2]

        n_episodes = exp_params.simulation_params['n_episodes']
        n_runs = exp_params.simulation_params['n_runs']
        metric_shape = (n_vals[0], n_vals[1], n_episodes)
        all_episodes = np.zeros((n_vals[0], n_vals[1], n_episodes, n_runs))
        metric = np.zeros(metric_shape)

        for i in range(n_vals[0]):
            val_i = param_study[param_keys[0]][i]
            for j in range(n_vals[1]):
                val_j = param_study[param_keys[1]][j]
                print(f"Testing {param_keys[0]}:{val_i} and {param_keys[1]}:{val_j}")
                agent_params, train_params = exp_params.copy_and_update_params(
                    param_keys, [val_i, val_j]
                )

                agent = factory.get(agent_params, train_params)
                run_steps = experiment_loop(
                    env, agent, render=render_mode,
                    **exp_params.simulation_params
                )
                avg_steps = np.mean(run_steps, axis=1)
                metric[i, j, :] = avg_steps
                all_episodes[i, j, :, :] = run_steps

        param_vals = [v for v in param_study.values()]
        joblib.dump({
            "all_episodes": all_episodes,
            "metric": metric,
            "keys": param_keys,
            "values": param_vals,
            "exp_params": exp_params.to_dict(),
        }, f"param_study_{date_time}.joblib")
        # idx= 0; plt.semilogy(np.arange(0, 60), metric[ :, idx, :].T); plt.legend(param_vals[0]); plt.title(f'Weight Decay {param_vals[1][idx]:0.1e}'); plt.ylim(bottom=100); plt.show()
    else:
        # Just a single evaluation
        agent = factory.get(
            exp_params.agent_params,
            exp_params.train_params,
        )
        run_steps = experiment_loop(
            env, agent, render=render_mode,
            **exp_params.simulation_params
        )

        avg_steps = np.mean(run_steps, axis=1)

        plt.semilogy(avg_steps);
        ax = plt.gca();
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%d"));
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"));
        plt.grid(which="both");
        plt.xlabel('Episode #');
        plt.ylim(bottom=100);
        plt.ylabel('Avg # of Steps to Reach Goal');
        plt.show()

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
