from contextlib import contextmanager
import multiprocessing
import os
import time

import torch

import cor_rl.agents
from cor_rl import environments
from cor_rl.agents.a2c import (
    InteractionResult,
    AdvantageActorCriticAgent
)


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

        if state is None:
            # Lost a life: episode restart
            break
        t += 1

    if terminated:
        value_est = 0.0
        state = None
    elif state is None:
        # Lost a life: episode restart. Take a few no-ops
        for _ in range(3):
            state, reward, terminated, _, _ = env.step(0)
        value_est = 0.0
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
    # # If the only step was losing a life, start the next life
    # # Since no information was provided
    # # TODO: Or should you have added a return of 0?
    # if state is None and len(results.rewards) == 0:
    #     results, state, terminated, frames = interact(
    #         env, agent, t_max=t_max, state=state, output_frames=output_frames
    #     )

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


def train_loop(global_agent: AdvantageActorCriticAgent, agents, envs,
               total_step_limit=10000, episode_limit=None,
               log_interval=1e9, solved_thresh=None, max_ep_steps=10000,
               steps_per_batch=10000, debug=False,
               avg_decay=0.95, seed=None):
    """
    This is a single threaded (serial) training loop with multiple agents
    """
    start_time = time.time()

    solved_thresh = solved_thresh or float('inf')
    episode_limit = episode_limit or 1e9
    total_steps = 0
    n_episodes = 0
    ep_steps, ep_reward = 0, 0
    avg_reward = 0
    solved = False

    n_threads = len(agents)
    states = [None] * n_threads
    print("")

    if seed:
        torch.manual_seed(seed)

    while total_steps < total_step_limit and n_episodes < episode_limit:
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
                agent, envs[t_idx], params, states[t_idx],
                t_max=steps_per_batch
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

    agent = cor_rl.agents.factory(agent_params, train_params)
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
                        avg_decay=0.95, debug=False, out_dir=None):
    """
    Training loop which sets up multiple worker threads which compute
    gradients in parallel.
    """
    start_time = time.time()

    solved_thresh = solved_thresh or float('inf')
    total_steps = total_episodes = 0
    ep_steps, ep_reward = [0]*n_workers, [0]*n_workers
    avg_reward = 0
    win_max_reward = max_reward = 0
    solved = False
    episode_limit = episode_limit or 1e9
    keep_training = True

    accumulate_grads = False

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    env_params = {
        'env_name': env_name,
        'seed': seed,
        'max_steps_per_episode': max_steps_per_episode,
    }

    # Seed and create the global agent
    if seed:
        torch.manual_seed(seed)
    global_agent = cor_rl.agents.factory(agent_params, train_params)
    global_agent.zero_grad()

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

                # TODO: Should this be adding the grads?
                if accumulate_grads:
                    global_agent.accumulate_grads(result['grads'])
                else:
                    global_agent.set_grads(result['grads'])
                    global_agent.backward()

                # Update counters and print out if necessary
                total_steps += result['n_steps']
                ep_reward[w_idx] += result['total_reward']

                if result['terminated']:
                    total_episodes += 1
                    last_reward = ep_reward[w_idx]
                    win_max_reward = max(last_reward, win_max_reward)
                    avg_reward = (
                        avg_decay * avg_reward +
                        (1.0 - avg_decay) * last_reward
                    )
                    solved = avg_reward > solved_thresh
                    ep_steps[w_idx] = ep_reward[w_idx] = 0
                    if (total_episodes % log_interval) == 0:
                        print(
                            f'Episode {total_episodes}\t'
                            f'Max reward: {win_max_reward:.2f}\t'
                            f'Average reward: {avg_reward:.2f}'
                        )
                        win_max_reward = 0
                        if out_dir:
                            out_file = f"ep_{total_episodes}.chkpt"
                            global_agent.checkpoint(
                                os.path.join(out_dir, out_file)
                            )
                if solved:
                    break
            if accumulate_grads and not solved:
                global_agent.backward()

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
