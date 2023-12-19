"""
Mountain car experiment runner
"""
import argparse
from copy import deepcopy
import datetime
import json
import os
import pprint
import time

import gymnasium as gym
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from cor_rl.utils import (
    write_gif,
    get_device
)
import q_agents
import ac_agents


DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "rl_results")





def experiment_loop(env, agent, out_dir, seed=101, n_runs=100, n_episodes=500,
                    eval_episodes={}, max_steps=[10000, 10000], verbose=True,
                    checkpoint_interval=None, output_gif=False):

    run_steps = np.zeros((n_episodes, n_runs))
    checkpoint_interval = checkpoint_interval or 1000000000

    os.makedirs(out_dir, exist_ok=1)

    if isinstance(max_steps, int):
        max_steps = [max_steps]*2

    step_limit = np.linspace(max_steps[0], max_steps[1], n_episodes)

    state, info = env.reset(seed=seed)
    for r_idx in range(n_runs):
        agent.checkpoint(os.path.join(out_dir, f"run{r_idx}_ep0.chkpt.pt"))
        for e_idx in range(n_episodes):
            frame_buffer = []
            s_idx = 0
            action_idx = agent.initialize(state)
            if output_gif:
                frame_buffer.append(env.render())
            trajectory = [state]
            actions = []
            st = time.time()
            state, reward, terminated, _, _ = env.step(action_idx)
            if output_gif:
                frame_buffer.append(env.render())
            while not terminated:
                action_idx = agent.step(reward, state)
                state, reward, terminated, _, _ = env.step(action_idx)
                if output_gif:
                    frame_buffer.append(env.render())

                trajectory.append(state)
                s_idx += 1
                actions.append(action_idx)

                # Pause if loop might be infinite
                if s_idx >= step_limit[e_idx]:
                    final_state = "aborted"
                    elap = round((time.time() - st)/60, 2)
                    break
            if terminated:
                final_state = "reached goal"
                agent.finish(reward)

            if output_gif:
                out_file = os.path.join(out_dir, f"run{r_idx+1}_ep{e_idx+1}.gif")
                write_gif(frame_buffer, out_file)

            elap = round((time.time() - st)/60, 2)

            if ((e_idx+1) % checkpoint_interval) == 0:
                agent.checkpoint(os.path.join(out_dir, f"run{r_idx}_ep{e_idx+1}.chkpt.pt"))

            # Record result and display if desired
            run_steps[e_idx, r_idx] = s_idx
            if verbose or (e_idx + 1) % 100 == 0:
                print(f"Run {r_idx+1} Episode {e_idx+1} {final_state} at step {s_idx} in {elap}min")
            if e_idx in eval_episodes:
                q_vals = agent.evaluate_q()

            state, info = env.reset()

        agent.checkpoint(os.path.join(out_dir, f"run{r_idx}_ep{e_idx+1}.chkpt.pt"))

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
        agent = constructor(
            self.n_actions, agent_params, train_params, device=self.device
        )
        load_file = agent_params.get('load_checkpoint', None)
        if load_file:
            agent.load(load_file)
        return agent


def plot_single_experiment_result(run_steps, save_loc=None, add_legend=False):
    avg_steps = np.mean(run_steps, axis=1);

    plt.semilogy(np.arange(run_steps.shape[0]), run_steps, linewidth=0.8);
    plt.semilogy(avg_steps, 'k', linewidth=2);
    ax = plt.gca();
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%d"));
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d"));
    plt.grid(which="both");
    plt.xlabel('Episode #');
    plt.ylim(bottom=100);
    plt.ylabel('# of Steps to Reach Goal');

    if add_legend:
        legend = [str(i) for i in range(run_steps.shape[1])]
        legend.append('Mean')
        plt.legend(legend)

    if save_loc is not None:
        plt.savefig(save_loc)
        print(f"Saved plot to {save_loc}")
    plt.show()


def plot_parameter_study(result, save_base=None):

    metric = result['metric'];
    param_vals = result['values'];
    n_episodes = metric.shape[-1];
    leg_param_name = result['keys'][0].rsplit(".", 1)[-1];

    for idx in range(metric.shape[1]):
        plt.figure(idx+1);
        plt.semilogy(np.arange(0, n_episodes), metric[:, idx, :].T);
        legend = [f'{leg_param_name} {v}' for v in param_vals[0]];
        plt.legend(legend);
        ax = plt.gca();
        plt.grid(which="both");
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%d"));
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"));
        param_name = result['keys'][1].rsplit(".", 1)[-1];
        plt.title(f'{param_name} {param_vals[1][idx]:0.1e}');
        plt.xlabel('Episode #');
        plt.ylim(bottom=100);
        plt.ylabel('Avg # of Steps to Reach Goal');

        if save_base is not None:
            save_loc = f'{save_base}_{idx}.png'
            plt.savefig(save_loc)
            print(f"Saved plot to {save_loc}")

    plt.show()


def plot_q_deltas(q_vals):

    plt.subplot(2, 2, 1);
    plt.imshow(q_vals[:, :, 0] - q_vals[:, :, 1]);
    plt.title("Q(s,a=0) - Q(s,a=1)")
    plt.colorbar();
    plt.subplot(2, 2, 2);
    plt.imshow(q_vals[:, :, 1] - q_vals[:, :, 2]);
    plt.title("Q(s,a=1) - Q(s,a=2)")
    plt.colorbar();
    plt.subplot(2, 2, 3);
    plt.imshow(q_vals[:, :, 0] - q_vals[:, :, 2]);
    plt.title("Q(s,a=0) - Q(s,a=2)")
    plt.colorbar();
    plt.subplot(2, 2, 4);
    plt.imshow(np.argmax(q_vals, axis=2));
    plt.colorbar();
    plt.title("$\pi(s)$")

    plt.show()


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
        help="Render the agent in the environment as it learns"
    )
    parser.add_argument(
        '-g', "--use_gpu", action="store_true",
        help="Turn on GPU acceleration. (Not good for single step algos)"
    )
    parser.add_argument(
        '-o', '--out_dir', default=DEFAULT_DIR,
        help="Where to store results"
    )
    parser.add_argument(
        '-f', "--forensic", type=str,
        help="Render the resulting value functions"
    )
    parser.add_argument(
        '-d', "--details", type=str,
        help="Render the details of a sweep: comma separated indexes"
    )

    # Get command line parameters and setup output directory
    cli_args = parser.parse_args()
    date_time = datetime.datetime.now().strftime('%Y_%b_%d_%H_%M')
    os.makedirs(cli_args.out_dir, exist_ok=True)

    # Parse and prepare experiment parameters
    if cli_args.json:
        with open(cli_args.json, 'r') as fp:
            json_params = json.load(fp)

        base_name = os.path.basename(cli_args.json).rsplit(".", 1)[0]
    else:
        json_params = {}
        base_name = date_time
    default_agent_type = "actor-critic" if "ac" in cli_args.json else "deepq"
    pprint.pprint(json_params)

    agent_type = json_params.pop("agent_type", None)
    if agent_type is None:
        print(f"WARNING! Agent Type no specified, using {default_agent_type}")
        agent_type = default_agent_type

    param_study = json_params.pop('param_study', {})
    exp_params = ExperimentParams(**json_params)

    # Setup the environment
    render_mode = "human" if cli_args.render else 'rgb_array'
    env = gym.make('MountainCar-v0', render_mode=render_mode)
    n_actions = env.action_space.n

    # Setup the agent
    device = get_device(cli_args.use_gpu)
    factory = AgentFactory(agent_type, n_actions, device=device)

    # Get details on a previous parameter study if desired
    if cli_args.details:

        sweep_data = joblib.load(cli_args.forensic)
        i, j = [int(i) for i in cli_args.details.split(",")]
        run_steps = sweep_data['all_episodes'][i, j, :, :]

        plot_single_experiment_result(run_steps, add_legend=True)

    # Do a forensic analysis of a run if desired
    elif cli_args.forensic:
        agent = factory.get(
            exp_params.agent_params,
            exp_params.train_params,
        )
        agent.load(cli_args.forensic)
        # Visualize the policy and/or value function
        agent.visualize()
        if cli_args.out_dir:
            # Run one time to output a GIF or something
            run_steps = experiment_loop(
                env, agent, cli_args.out_dir,
                **exp_params.simulation_params
            )
            # Visualize again to check policy stability
            agent.visualize()

    # Run a parameter study if desired
    elif param_study:
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
                    env, agent,
                    out_dir=os.path.join(cli_args.out_dir, f"sweep{i}_{j}"),
                    **exp_params.simulation_params
                )
                avg_steps = np.mean(run_steps, axis=1)
                metric[i, j, :] = avg_steps
                all_episodes[i, j, :, :] = run_steps

        param_vals = [v for v in param_study.values()]
        result = {
            "all_episodes": all_episodes,
            "metric": metric,
            "keys": param_keys,
            "values": param_vals,
            "exp_params": exp_params.to_dict(),
        }
        descrip = "param_study" if base_name == date_time else base_name
        out_file = os.path.join(cli_args.out_dir, f"{descrip}_{date_time}.joblib")
        joblib.dump(result, out_file)
        save_base = os.path.join(cli_args.out_dir, f"{descrip}_{date_time}_sweep")

        plot_parameter_study(result, save_base=save_base)
    else:
        # Just run a single evaluation
        agent = factory.get(
            exp_params.agent_params,
            exp_params.train_params,
        )
        run_steps = experiment_loop(
            env, agent, cli_args.out_dir,
            **exp_params.simulation_params
        )

        descrip = "result" if base_name == date_time else base_name
        save_loc = os.path.join(cli_args.out_dir, f"{descrip}_{date_time}.png")
        plot_single_experiment_result(run_steps, save_loc=save_loc)

    # Final debug for developer analysis
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
