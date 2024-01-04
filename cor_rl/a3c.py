from contextlib import contextmanager
import datetime
import math
import multiprocessing
import os
import select
import time

import mlflow
import torch

import cor_rl.agents
from cor_rl import environments
from cor_rl.agents.a2c import (
    InteractionResult,
    AdvantageActorCriticAgent
)

MLFLOW_URI = "http://127.0.0.1:8888"
mlflow.set_tracking_uri(uri=MLFLOW_URI)


def interact(env, agent, t_max=5, state=None, output_frames=False,
             break_on_lost_life=True):
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
            if break_on_lost_life:
                # Lost a life: episode restart
                break
            else:
                # Do a no-op to get some data
                state, reward, terminated, _, _ = env.step(0)
                results.rewards.append(reward)
                results.values.append(0.0)
                results.log_probs.append(0.0)
                results.entropies.append(0.0)
        t += 1

    if terminated:
        value_est = torch.tensor([0.0])
        state = None
    elif state is None:
        # Lost a life: episode restart. Take a few no-ops
        for _ in range(3):
            state, reward, terminated, _, _ = env.step(0)
        value_est = torch.tensor([0.0])
    else:
        # Get an estimate of the value of the final state
        with torch.no_grad():
            action_idx, value_est, _, _ = agent.select_action(state)
        value_est = value_est.item()

    results.values.append(value_est)

    return results, state, terminated, frame_buffer


def agent_env_task(agent, env, parameters, state, t_max=5,
                   output_frames=False, eval_mode=False):

    if parameters is not None:
        agent.set_parameters(parameters)

    agent.zero_grad()

    results, state, terminated, frames = interact(
        env, agent, t_max=t_max, state=state, output_frames=output_frames,
        break_on_lost_life=not eval_mode
    )
    # # If the only step was losing a life, start the next life
    # # Since no information was provided
    # # TODO: Or should you have added a return of 0?
    # if state is None and len(results.rewards) == 0:
    #     results, state, terminated, frames = interact(
    #         env, agent, t_max=t_max, state=state, output_frames=output_frames
    #     )
    metrics = {}
    # This will run back prop
    if parameters is None:
        grads = None
        if not eval_mode:
            loss = agent.calculate_loss(results)
            metrics = agent.backward(loss)

    else:
        grads, loss = agent.calc_and_get_grads(results)

    output = {
        'grads': grads,
        'state': state,
        'total_reward': sum(results.rewards),
        'rewards': results.rewards,
        'terminated': terminated,
        'n_steps': len(results.rewards),
    }
    output.update(metrics)

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
    avg_reward = max_reward = 0
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
            max_reward = max(max_reward, ep_reward)
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
                    elap = (time.time() - start_time) / 60
                    print(
                        f'Episode {n_episodes}\tMax reward: {max_reward:.2f}\t'
                        f'Average reward: {avg_reward:.2f}\t Time: {elap:0.1f}min'
                    )
                    max_reward = 0
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


class PipeMock:
    """
    Objects to crudely mock the Pipe API to implement serial processing
    """

    def __init__(self, worker):
        self.worker = worker
        self.queue = []

    def send(self, payload):
        result = self.worker.handle_task(payload)
        self.queue.append(result)

    def recv(self):
        result = self.queue.pop(0)
        return result


@contextmanager
def serial_workers(n_workers, worker_func, worker_args, agent=None):
    """
    Implements serial workers via mocking the pipe API
    """
    # Setup training workers
    workers = [
        Worker(i, *worker_args, agent=agent) for i in range(n_workers)
    ]
    mock_pipes = [PipeMock(worker) for worker in workers]

    # Add the evaluation worker
    workers.append(Worker(0, *worker_args, eval_mode=True))
    mock_pipes.append(PipeMock(workers[-1]))

    try:
        yield mock_pipes
    finally:
        pass


@contextmanager
def piped_workers(n_workers, worker_func, worker_args, agent=None):
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
            args=[i, child_conn, *worker_args],
            kwargs={"agent": agent},
        )
        worker_processes.append(worker_process)
        worker_process.start()

    parent_conn, child_conn = multiprocessing.Pipe()
    parent_conns.append(parent_conn)
    child_conns.append(child_conn)
    eval_process = multiprocessing.Process(
        target=worker_func,
        args=[0, child_conn, *worker_args],
        kwargs={'eval_mode': True},
    )
    worker_processes.append(eval_process)
    eval_process.start()

    try:
        yield parent_conns
    finally:
        for i in range(n_workers+1):
            worker_processes[i].terminate()
            parent_conns[i].close()
            child_conns[i].close()


class Worker:

    def __init__(self, task_id, agent_params, train_params, env_params,
                 agent=None, eval_mode=False):
        self.eval_mode = eval_mode
        self.shared_mode = agent is not None
        self.task_id = task_id
        agent_params = dict(agent_params)
        env_params = dict(env_params)
        if eval_mode:
            # Turn off reward clipping
            env_params.pop('reward_clip', None)
            agent_params['reward_clip'] = None

        env_name = env_params.pop('env_name')
        self.seed = env_params.pop('seed', 8888)
        self.max_steps_per_episode = env_params.pop('max_steps_per_episode', 1e9)
        self.env = environments.factory.get(env_name, **env_params)
        self.ep_steps = 0
        if self.seed:
            self.task_seed = self.seed + task_id * 10
            self.state, info = self.env.reset(seed=self.task_seed)
            torch.manual_seed(self.task_seed)
        if not eval_mode:
            self.state = None  # Force reset to match previous work

        if not self.shared_mode:
            self.agent = cor_rl.agents.factory(agent_params, train_params)
        else:
            self.agent = agent
        torch.manual_seed(self.task_seed)  # Reset seed to match previous work

    def handle_task(self, task):
        task_type = task.get('type', '')

        if task_type == 'train':
            # Allow controller to reset the environment
            if task.get('reset', False):
                self.state = None

            # Train the agent for a few steps
            params = None if self.shared_mode else task['params']
            result = agent_env_task(
                self.agent, self.env, params, self.state,
                t_max=task['max_steps']
            )
            # Update the state for next time
            self.state = result['state']
            # Keep track of steps for this episode and end if too many
            self.ep_steps += result['n_steps']
            if self.ep_steps >= self.max_steps_per_episode:
                self.state = None
                result['terminated'] = True

            # Restart counter if terminated
            if result['terminated']:
                self.ep_steps = 0

            # Send result to parent process
            return result
        elif task_type == 'eval':
            if not self.eval_mode:
                raise ValueError("You should be in eval mode!")

            self.agent.set_parameters(task['params'])
            if task.get('save_file'):
                self.agent.checkpoint(task['save_file'])

            # Play N games and average the score
            n_games = task.get('n_games', 8)
            scores = []
            with torch.no_grad():
                for g_idx in range(n_games):
                    result = agent_env_task(
                        self.agent, self.env, None, state=self.state,
                        t_max=100000, eval_mode=True
                    )
                    self.state = None
                    scores.append(result['total_reward'])

            avg_score = sum(scores) / n_games
            std_scores = sum((s - avg_score)**2 for s in scores) / n_games
            result = {
                'scores': scores,
                'avg_score': avg_score,
                'std_score': math.sqrt(std_scores),
                'reward_clip': self.agent.reward_clip,
                'rewards': result['rewards'],
            }
            return result
        elif task_type == "save":
            self.agent.set_parameters(task['params'])
            self.agent.checkpoint(task['file_name'])

            return task['file_name']

        elif task_type == 'params':
            params = self.agent.get_parameters()
            params['seed'] = [self.task_seed, self.seed]
            params['state'] = [] if self.state is None else self.state.tolist()
            return params
        elif task_type == 'state':
            return [] if self.state is None else self.state.tolist()
        elif task_type == 'STOP':
            return "FINISH HIM!"
        else:
            return "NO TYPE"


def worker_thread_new(task_id, conn, agent_params, train_params, env_params,
                      agent=None, eval_mode=False):

    worker = Worker(
        task_id, agent_params, train_params, env_params,
        agent=agent, eval_mode=eval_mode
    )

    while True:
        task = conn.recv()
        if task.get('type', '') == 'STOP':
            conn.send("FINISH HIM!")
            break
        result = worker.handle_task(task)
        conn.send(result)


def worker_thread(task_id, conn, agent_params, train_params, env_params,
                  agent=None, eval_mode=False):
    """
    This is the thread which trains an agent based on messages from the parent.
    It receives the current global model parameters and
    returns gradient updates to apply to the global model.

    Eval mode turns off gradients and reward clipping
    """

    shared_mode = agent is not None

    if eval_mode:
        agent_params = dict(agent_params)
        env_params = dict(env_params)
        # Turn off reward clipping
        env_params.pop('reward_clip', None)
        agent_params['reward_clip'] = None

    env_name = env_params.pop('env_name')
    seed = env_params.pop('seed', 8888)
    max_steps_per_episode = env_params.pop('max_steps_per_episode', 1e9)
    env = environments.factory.get(env_name, **env_params)
    ep_steps = 0
    if seed:
        task_seed = seed + task_id * 10
        state, info = env.reset(seed=task_seed)
        torch.manual_seed(task_seed)
    if not eval_mode:
        state = None  # Force reset to match previous work

    if not shared_mode:
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
            params = None if shared_mode else task['params']
            result = agent_env_task(
                agent, env, params, state,
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
        elif task_type == 'eval':
            if not eval_mode:
                raise ValueError("You should be in eval mode!")

            agent.set_parameters(task['params'])
            if task.get('save_file'):
                # torch.save(agent.state_dict(), task['save_file'])
                agent.checkpoint(task['save_file'])

            # Play N games and average the score
            n_games = task.get('n_games', 8)
            scores = []
            with torch.no_grad():
                for g_idx in range(n_games):
                    result = agent_env_task(
                        agent, env, None, state=state, t_max=100000,
                        eval_mode=True
                    )
                    state = None
                    scores.append(result['total_reward'])

            avg_score = sum(scores) / n_games
            std_scores = sum((s - avg_score)**2 for s in scores) / n_games
            result = {
                'scores': scores,
                'avg_score': avg_score,
                'std_score': math.sqrt(std_scores),
                'reward_clip': agent.reward_clip,
                'rewards': result['rewards'],
            }
            conn.send(result)
        elif task_type == "save":
            agent.set_parameters(task['params'])
            # torch.save(agent.state_dict(), task['file_name'])
            agent.checkpoint(task['file_name'])

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
        else:
            conn.send("NO TYPE")


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


def log_metrics(metrics, step):
    for metric_name in ('grad_norm', 'loss'):
        mlflow.log_metric(
            metric_name, metrics[metric_name],
            step=step, synchronous=False
        )


def train_loop_parallel(n_workers, agent_params, train_params, env_params,
                        steps_per_batch=5, total_step_limit=10000,
                        episode_limit=None, max_steps_per_episode=10000,
                        solved_thresh=None, log_interval=1e9, seed=8888,
                        avg_decay=0.95, debug=False, out_dir=None,
                        eval_interval=None, accumulate_grads=False,
                        experiment_name=None, load_file=None, save_interval=None,
                        use_mlflow=True, serial=False):
    """
    Training loop which sets up multiple worker threads which compute
    gradients in parallel.

    eval_interval: in epochs (4M frames)
    save_interval: in epochs
    """

    experiment_name = experiment_name or datetime.datetime.now().strftime("%Y_%b_%d_H%H_%M")
    if use_mlflow:
        mlflow.set_experiment(experiment_name)
        agent_params['experiment_name'] = experiment_name
        mlflow.start_run()

    save_interval = save_interval or float('inf')
    eval_interval = eval_interval or float('inf')
    last_eval_epoch = -eval_interval if eval_interval < float('inf') else 0.0
    last_save = 0
    eval_in_flight = False

    solved_thresh = solved_thresh or float('inf')
    total_steps = total_episodes = 0
    ep_steps, ep_reward = [0]*n_workers, [0]*n_workers
    avg_reward = 0
    win_max_reward = max_reward = 0
    solved = False
    episode_limit = episode_limit or 1e9
    keep_training = True
    metric_step_rate = 500

    # If global_agent is shared between threads rather that copied
    shared_mode = True

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if isinstance(env_params, str):
        env_params = {'env_name': env_params}
    env_params['seed'] = seed
    env_params['max_steps_per_episode'] = max_steps_per_episode

    # Seed and create the global agent
    if seed:
        torch.manual_seed(seed)
    global_agent = cor_rl.agents.factory(agent_params, train_params)
    if load_file:
        print(f"Loading agent from {load_file}!")
        global_agent.load(load_file)
    if use_mlflow:
        mlflow.log_params(global_agent.params())
    global_agent.zero_grad()
    if shared_mode:
        global_agent.share_memory()
        pipe_agent = global_agent
    else:
        pipe_agent = None

    worker_args = [agent_params, train_params, env_params]

    if serial:
        context_func = serial_workers
    else:
        context_func = piped_workers

    with context_func(n_workers, worker_thread_new, worker_args, agent=pipe_agent) as msg_pipes:
        start_time = time.time()
        while keep_training:
            if not shared_mode:
                params = global_agent.get_parameters()
            else:
                params = None
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

            elap_time = (time.time() - start_time) / 60
            n_epochs = total_steps / 4e6
            # do_eval = (elap_time - last_eval_time) > eval_interval
            do_eval = (n_epochs - last_eval_epoch) >= eval_interval
            if do_eval:
                params = global_agent.get_parameters()
                payload = {
                    'type': 'eval', 'params': params
                }
                eval_steps = total_steps
                do_save = (n_epochs - last_save) > save_interval
                if do_save:
                    last_save = round(n_epochs, 2)
                    payload['save_file'] = os.path.join(
                        out_dir, f'{experiment_name}_epoch{last_save}.pt'
                    )

                # TODO: Make these pipes synchronous classes
                # That store results on send and receive
                msg_pipes[n_workers].send(payload)
                eval_in_flight = True
                last_eval_epoch = round(n_epochs, 3)

            # Get the result from each worker and update the model
            for w_idx in range(n_workers):
                result = msg_pipes[w_idx].recv()

                if use_mlflow and total_steps % metric_step_rate == 0:
                    log_metrics(result, step=total_steps)

                if debug:
                    _show_grads(result)

                if not shared_mode:
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
                    if use_mlflow:
                        mlflow.log_metric(
                            'ep_reward', last_reward,
                            step=total_episodes, synchronous=False
                        )
                    win_max_reward = max(last_reward, win_max_reward)
                    avg_reward = (
                        avg_decay * avg_reward +
                        (1.0 - avg_decay) * last_reward
                    )
                    solved = avg_reward > solved_thresh
                    ep_steps[w_idx] = ep_reward[w_idx] = 0
                    if (total_episodes % log_interval) == 0:
                        elap = (time.time() - start_time) / 60
                        print(
                            f'Episode {total_episodes}\t'
                            f'Average reward: {avg_reward:.2f}\t'
                            f'Max reward: {win_max_reward:.2f}\t'
                            f'Time: {elap:0.1f}min'
                        )
                        win_max_reward = 0
                        # if out_dir:
                        #     out_file = f"ep_{total_episodes}.chkpt"
                        #     global_agent.checkpoint(
                        #         os.path.join(out_dir, out_file)
                        #     )
                if solved:
                    break
            if (not shared_mode and accumulate_grads) and not solved:
                global_agent.backward()

            if eval_in_flight:

                eval_conn = msg_pipes[n_workers]
                readable_fds, _, _ = select.select(
                    [eval_conn], [], [], 0  # 0 is timeout (no blocking)
                )
                if eval_conn in readable_fds:
                    result = eval_conn.recv()
                    print(
                        f"Epoch {eval_steps / 4e6:0.3f}\t"
                        f"Average score: {result['avg_score']:0.1f}\t"
                        f"Std score: {result['std_score']:0.1f}  \t"
                        f"Time: {last_eval_epoch:0.1f}min"
                    )
                    if use_mlflow:
                        mlflow.log_metric(
                            'avg_score', result['avg_score'],
                            step=eval_steps, synchronous=False
                        )
                    eval_in_flight = False

            keep_training = (
                not solved and
                total_steps < total_step_limit and
                total_episodes < episode_limit
            )

    if use_mlflow:
        mlflow.end_run()

    if solved:
        print(
            f'Episode {total_episodes}\tLast reward: {last_reward:.2f}\t'
            f'Average reward: {avg_reward:.2f}'
        )
        print(f"PROBLEM SOLVED in {time.time() - start_time:0.1f} sec!")
    else:
        print(f"Aborted after {time.time() - start_time:0.1f}sec")

    return global_agent, solved
