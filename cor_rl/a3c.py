from contextlib import contextmanager
from contextlib import nullcontext
import datetime
import math
import torch.multiprocessing as mp
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
             break_on_lost_life=False, lock=None):
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
        action_idx, value_est, entropy, log_prob = agent.select_action(state, lock=lock)
        state, reward, terminated, _, _ = env.step(action_idx)

        results.rewards.append(reward)
        results.values.append(value_est)
        results.log_probs.append(log_prob)
        results.entropies.append(entropy)

        if output_frames:
            frame_buffer.append(env.render())

        # if state is None:
        #     if break_on_lost_life:
        #         # Lost a life: episode restart
        #         break
        #     else:
        #         # Do a no-op to get some data and avoid triggering
        #         # environment reset. This is for eval mode
        #         state, reward, terminated_no_op, _, _ = env.step(0)
        #         results.rewards.append(reward)
        #         results.values.append(0.0)
        #         results.log_probs.append(0.0)
        #         results.entropies.append(0.0)
        t += 1

    if terminated:
        # Having this be a float and not tensor is important for stability
        value_est = 0.0
        state = None
    elif state is None:
        # Lost a life: episode restart. Take a no-op so state is not None
        # so that it doesn't trigger an environment reset
        for _ in range(1):
            state, reward, terminated_no_op, _, _ = env.step(0)
            # Just in case there were points scored after death
            results.rewards[-1] += reward

        # Having this be a float and not tensor is important for stability
        value_est = 0.0
    else:
        # Get an estimate of the value of the final state
        with torch.no_grad():
            action_idx, value_est, _, _ = agent.select_action(state, lock=lock)
        value_est = value_est.item()

    results.values.append(value_est)

    return results, state, terminated, frame_buffer


def agent_env_task(agent: AdvantageActorCriticAgent, shared_agent: AdvantageActorCriticAgent,
                   shared_opt, env, state, t_max=5,
                   output_frames=False, eval_mode=False, lock=nullcontext()):
    # lock = NoOpContextManager()

    # if parameters is not None:
    #     agent.set_parameters(parameters)
    agent.set_parameters(shared_agent.get_parameters())

    agent.zero_grad()

    results, state, terminated, frames = interact(
        env, agent, t_max=t_max, state=state, output_frames=output_frames,
        break_on_lost_life=not eval_mode,
        lock=None,  # Do not typically lock on forward pass
    )
    metrics = {}
    # # This will run back prop
    # if parameters is None:
    #     grads = None
    #     if not eval_mode:
    #         # with lock:
    #         loss = agent.calculate_loss(results)
    #         with lock:  # TODO: Probably only need to do here.
    #             metrics = agent.backward(loss)

    # else:
    #     grads, loss = agent.calc_and_get_grads(results)
    grads = None
    if not eval_mode:
        # shared_opt.zero_grad()
        loss, norm_val = agent.calc_loss_and_backprop(results)
        with lock:
            shared_opt.zero_grad(set_to_none=True)
            agent.sync_grads(shared_agent)
            shared_opt.step()

        metrics['loss'] = loss.item()
        metrics['grad_norm'] = norm_val.item()

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
def serial_workers(n_workers, worker_func, worker_args, agent=None, render=False, **kwargs):
    """
    Implements serial workers via mocking the pipe API
    """
    # Setup training workers
    # render_mode='human' if i == 0 else None
    workers = [
        Worker(i, *worker_args, shared_agent=agent, shared_opt=agent.optimizer, render=render)
        for i in range(n_workers)
    ]
    mock_pipes = [PipeMock(worker) for worker in workers]

    # Add the evaluation worker
    if not kwargs.get('repro_mode', False):
        workers.append(Worker(0, *worker_args, shared_agent=agent, eval_mode=True))
        mock_pipes.append(PipeMock(workers[-1]))

    try:
        yield mock_pipes
    finally:
        pass


@contextmanager
def piped_workers(n_workers, worker_func, worker_args, agent=None, render=False, use_lock=True, **kwargs):
    """
    Context manager to manage a set of training worker threads.

    Pipes and processes are automatically closed and terminated.
    """

    parent_conns, child_conns = [], []
    worker_processes = []

    lock = mp.Lock() if use_lock else None

    for i in range(n_workers):
        parent_conn, child_conn = mp.Pipe()
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)
        worker_process = mp.Process(
            target=worker_func,
            args=[i, child_conn, *worker_args],
            kwargs={"agent": agent, "render": render, "lock": lock},
        )
        worker_processes.append(worker_process)
        worker_process.start()

    parent_conn, child_conn = mp.Pipe()
    parent_conns.append(parent_conn)
    child_conns.append(child_conn)
    eval_process = mp.Process(
        target=worker_func,
        args=[0, child_conn, *worker_args],
        kwargs={'eval_mode': True, "agent": agent},
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
                 shared_agent, shared_opt=None, eval_mode=False, render=False, lock=None):
        self.eval_mode = eval_mode
        # self.shared_mode = agent is not None
        self.task_id = task_id
        self.lock = lock or nullcontext()
        self.epoch_steps = 4e6
        agent_params = dict(agent_params)
        env_params = dict(env_params)
        if eval_mode:
            # Turn off reward clipping
            env_params.pop('reward_clip', None)
            agent_params['reward_clip'] = None

        if render and task_id == 0:
            env_params['render_mode'] = 'human'

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

        # if not self.shared_mode:
        #     self.agent = cor_rl.agents.factory(agent_params, train_params)
        # else:
        #     self.agent = agent
        self.agent = cor_rl.agents.factory(agent_params, train_params)
        self.shared_agent = shared_agent
        self.shared_opt = shared_opt

        # Turning off to match Cart Pole Reference
        # torch.manual_seed(self.task_seed)  # Reset seed to match previous work

    def handle_task(self, task):
        task_type = task.get('type', '')

        if task_type == 'train':
            # Allow controller to reset the environment
            if task.get('reset', False):
                self.state = None

            # Train the agent for a few steps
            # params = None if self.shared_mode else task['params']
            result = agent_env_task(
                self.agent, self.shared_agent, self.shared_opt,
                self.env, self.state,
                t_max=task['max_steps'], lock=self.lock
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
                        self.agent, self.shared_agent, self.shared_opt,
                        self.env, state=self.state,
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

    def reset_metrics(self):
        self.metrics = dict(
            ep_score=0,
            ep_loss=0,
            ep_grad_norm=0.0,
            max_loss=float('-inf'),
            max_grad_norm=float('-inf'),
            ep_steps=0,
            ep_batches=0
        )

    def setup_logging(self, worker_params):
        self.mlflow_run_id = worker_params.get('mlflow_run_id')
        self.use_mlflow = self.mlflow_run_id is not None
        self.metric_log_interval = worker_params.get('metric_log_interval', 2)
        self.print_interval = worker_params.get('print_interval', 10)

        if self.use_mlflow:
            mlflow.set_experiment(worker_params['experiment_name'])
            mlflow.start_run(self.mlflow_run_id, nested=True)
            if self.task_id == 0:
                mlflow.log_params(self.agent.params())

        self.start_time = time.time()

    def teardown_logging(self):
        # if self.use_mlflow:
        #     mlflow.end_run()
        pass

    def log_metrics(self, episode_num, shared_info):
        avg_score = shared_info['avg_score'].item()
        if (episode_num % self.print_interval) == 0:
            elap_time = (time.time() - self.start_time) / 60.0
            epoch = shared_info['total_steps'].item() / self.epoch_steps
            print(
                f"Epoch: {epoch:0.2f}\t"
                f"Episode {episode_num:.0f}\t"
                f"Score: {avg_score:0.2f}\t"
                f"Time: {elap_time:0.2f}min\t"
            )
        if episode_num % self.metric_log_interval == 0:
            if self.use_mlflow:
                metrics = {
                    'running_avg_score': avg_score,
                    'running_avg_loss': shared_info['avg_loss'].item(),
                    'ep_avg_loss': self.metrics['ep_loss'] / self.metrics['ep_steps'],
                    'ep_avg_grad_norm': self.metrics['ep_grad_norm'] / self.metrics['ep_batches'],
                    'ep_max_loss': self.metrics['max_loss'],
                    'ep_max_grad_norm': self.metrics['max_grad_norm']
                }
                mlflow.log_metrics(
                    metrics, step=int(episode_num), synchronous=False
                )

    def continuously_train(self, shared_info, worker_params={}):
        """
            shared_info:
                total_steps
                total_episodes
                avg_score
                avg_loss
        """
        keep_training = True
        solved = False

        self.reset_metrics()

        metric_decay = worker_params.get('metric_decay', 0.95)
        solved_thresh = worker_params.get('solved_thresh') or float('inf')

        max_steps_per_batch = worker_params.get('max_steps_per_batch', 5)
        max_steps = worker_params.get('max_steps') or 200e6
        max_episodes = worker_params.get('max_episodes') or 1e9
        # Setup logging (mlflow experiment, etc)
        self.setup_logging(worker_params)

        while keep_training:
            self.agent.set_parameters(self.shared_agent.get_parameters())

            self.agent.zero_grad(set_to_none=True)

            results, state, terminated, frames = interact(
                self.env, self.agent, t_max=max_steps_per_batch,
                state=self.state, output_frames=False,
                break_on_lost_life=True,
            )
            self.state = state
            loss, norm_val = self.agent.calc_loss_and_backprop(results)
            # with lock:
            if shared_info['solved'].item() == 0:
                with self.lock:
                    self.shared_opt.zero_grad(set_to_none=True)
                    self.agent.sync_grads(self.shared_agent)
                    self.shared_opt.step()
            else:
                break
            # end with lock

            n_steps = len(results.rewards)
            self.metrics['ep_score'] += sum(results.rewards)
            self.metrics['ep_loss'] += loss.item()
            self.metrics['ep_grad_norm'] += norm_val.item()
            self.metrics['max_loss'] = max(self.metrics['max_loss'], loss.item())
            self.metrics['max_grad_norm'] = max(
                self.metrics['max_grad_norm'], norm_val.item()
            )
            shared_info['total_steps'] += n_steps
            self.metrics['ep_batches'] += 1
            self.metrics['ep_steps'] += n_steps
            if self.metrics['ep_steps'] >= self.max_steps_per_episode:
                self.state = None
                terminated = True

            if terminated:
                total_eps = shared_info['total_episodes'].item()
                shared_info['total_episodes'] += 1

                factor = 1 if total_eps == 0 else 1 - metric_decay
                shared_info['avg_score'].mul_(1-factor).add_(
                    factor * self.metrics['ep_score']
                )
                loss_per_step = self.metrics['ep_loss'] / self.metrics['ep_steps']
                shared_info['avg_loss'].mul_(1-factor).add_(factor * loss_per_step)

                self.log_metrics(total_eps+1, shared_info)
                solved = shared_info['avg_score'] >= solved_thresh
                if solved:
                    shared_info['solved'] += 1.0
                    elap_time = (time.time() - self.start_time) / 60.0
                    print(
                        f"SOLVED!\nEpisode {shared_info['total_episodes'].item():.0f}\t"
                        f"Avg Score: {shared_info['avg_score'].item():.2f}\t"
                        f"Last score: {self.metrics['ep_score']:.1f}\t"
                        f"Time: {elap_time:0.2f}min"
                    )
                    break
                self.reset_metrics()

            solved = solved or shared_info['solved'].item() > 0
            keep_training = (
                (not solved) and
                shared_info['total_episodes'] < max_episodes and
                shared_info['total_steps'] < max_steps
            )

        self.teardown_logging()


def worker_thread_new(task_id, conn, agent_params, train_params, env_params,
                      agent=None, eval_mode=False, render=False, lock=None):

    worker = Worker(
        task_id, agent_params, train_params, env_params,
        shared_agent=agent, shared_opt=agent.optimizer,
        eval_mode=eval_mode, render=render, lock=lock
    )

    while True:
        task = conn.recv()
        if task.get('type', '') == 'STOP':
            conn.send("FINISH HIM!")
            break
        result = worker.handle_task(task)
        conn.send(result)


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
                        use_mlflow=False, serial=False, shared_mode=True, render=False,
                        use_lock=True, repro_mode=False):
    """
    Training loop which sets up multiple worker threads which compute
    gradients in parallel.

    eval_interval: in epochs (4M frames)
    save_interval: in epochs
    shared_mode: If global_agent is shared between threads rather that copied
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
    avg_reward = None
    win_max_reward = max_reward = 0
    solved = False
    episode_limit = episode_limit or 1e9
    keep_training = True
    metric_step_rate = 2000
    last_flow_log = -metric_step_rate

    if repro_mode:
        avg_reward = 10

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if isinstance(env_params, str):
        env_params = {'env_name': env_params}
    env_params['seed'] = seed
    env_params['max_steps_per_episode'] = max_steps_per_episode

    # Seed and create the global agent
    if seed:
        torch.manual_seed(seed)
    g_agent_params = dict(agent_params)
    if shared_mode:
        g_agent_params["shared"] = True
    global_agent = cor_rl.agents.factory(g_agent_params, train_params)
    if load_file:
        print(f"Loading agent from {load_file}!")
        global_agent.load(load_file)
    if use_mlflow:
        mlflow.log_params(global_agent.params())
    global_agent.zero_grad()
    if shared_mode:
        global_agent.share_memory()

    worker_args = [agent_params, train_params, env_params]

    if serial:
        context_func = serial_workers
    else:
        context_func = piped_workers

    use_lock = use_lock and (not serial)

    with context_func(n_workers, worker_thread_new, worker_args,
                      agent=global_agent, render=render, use_lock=use_lock,
                      repro_mode=repro_mode) as msg_pipes:
        # Set seed to have same action selection in serial (debugging) mode
        if seed and not repro_mode:
            torch.manual_seed(seed)
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
                # For serial repro purposes
                # if not shared_mode:
                #     result = msg_pipes[w_idx].queue[0]
                #     global_agent.set_grads(result['grads'])
                #     global_agent.backward()

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

                msg_pipes[n_workers].send(payload)
                eval_in_flight = True
                last_eval_epoch = round(n_epochs, 3)

            # Get the result from each worker and update the model
            for w_idx in range(n_workers):
                result = msg_pipes[w_idx].recv()
                # if result['loss'] > 100:
                #     import ipdb; ipdb.set_trace()
                do_log = (
                    result['loss'] > 1000 or
                    (total_steps - last_flow_log) >= metric_step_rate
                )
                if use_mlflow and do_log:
                    last_flow_log = total_steps
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
                    if avg_reward is None:
                        avg_reward = last_reward
                    avg_reward = (
                        avg_decay * avg_reward +
                        (1.0 - avg_decay) * last_reward
                    )
                    solved = avg_reward > solved_thresh
                    ep_steps[w_idx] = 0
                    ep_reward[w_idx] = 0
                    if (total_episodes % log_interval) == 0:

                        elap = (time.time() - start_time) / 60
                        if repro_mode:
                            print(
                                f'Episode {total_episodes}\t'
                                f'Last reward: {last_reward:.2f}\t'
                                f'Average reward: {avg_reward:.2f}\t'
                                f'Time: {elap:0.1f}min'
                            )
                        else:
                            print(
                                f'Episode {total_episodes}\t'
                                f'Steps: {total_steps}\t'
                                f'Average reward: {avg_reward:.2f}\t'
                                f'Max reward: {win_max_reward:.2f}\t'
                                f'Time: {elap:0.1f}min'
                            )
                        win_max_reward = 0
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
                        f"Std score: {result['std_score']:0.1f} \t"
                        f"Max score: {max(result['scores']):0.1f}  \t"
                        f"Time: {last_eval_epoch:0.1f}min"
                    )
                    if use_mlflow:
                        mlflow.log_metric(
                            'avg_score', result['avg_score'],
                            step=eval_steps, synchronous=False
                        )
                        mlflow.log_metric(
                            'max_score', max(result['score']),
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
            f'Episode {total_episodes}\t'
            f'Average reward: {avg_reward:.2f}\t'
            f'Last reward: {last_reward:.2f}\t'
        )
        print(f"PROBLEM SOLVED in {time.time() - start_time:0.1f} sec!")
    else:
        print(f"Aborted after {time.time() - start_time:0.1f}sec")

    return global_agent, solved


def continuous_worker_thread(task_id, agent_params, train_params, env_params,
                             shared_agent, shared_info, worker_params, lock=None,
                             worker=None):
    if worker is None:
        worker = Worker(
            task_id, agent_params, train_params, env_params,
            shared_agent=shared_agent, shared_opt=shared_agent.optimizer,
            eval_mode=False, render=False, lock=lock
        )

    worker.continuously_train(shared_info, worker_params)

    return worker

def train_loop_continuous(n_workers, agent_params, train_params, env_params,
                          steps_per_batch=5, total_step_limit=10000,
                          episode_limit=None, max_steps_per_episode=10000,
                          solved_thresh=None, log_interval=1e9, seed=8888,
                          avg_decay=0.95, debug=False, out_dir=None,
                          eval_interval=None, accumulate_grads=False,
                          experiment_name=None, load_file=None, save_interval=None,
                          use_mlflow=False, serial=False, shared_mode=True, render=False,
                          use_lock=True, repro_mode=False):

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if isinstance(env_params, str):
        env_params = {'env_name': env_params}
    env_params['seed'] = seed
    env_params['max_steps_per_episode'] = max_steps_per_episode

    # Seed and create the global agent
    if seed:
        torch.manual_seed(seed)
    g_agent_params = dict(agent_params)
    if shared_mode:
        g_agent_params["shared"] = True
    global_agent = cor_rl.agents.factory(g_agent_params, train_params)
    if load_file:
        print(f"Loading agent from {load_file}!")
        global_agent.load(load_file)
    global_agent.zero_grad()
    if shared_mode:
        global_agent.share_memory()

    # TODO: Move this to worker
    # if use_mlflow:
    #     mlflow.log_params(global_agent.params())

    lock = mp.Lock() if use_lock else None

    info_fields = [
        'total_steps', 'total_episodes', 'avg_score', 'avg_loss', 'solved',
    ]
    shared_info = {
        k: torch.DoubleTensor([0]).share_memory_() for k in info_fields
    }
    if repro_mode:
        shared_info['avg_score'] += 10.0

    experiment_name = experiment_name or datetime.datetime.now().strftime("%Y_%b_%d_H%H_%M")
    mlflow_run_id = None
    if use_mlflow:
        mlflow.set_experiment(experiment_name)
        active_run = mlflow.start_run()
        mlflow_run_id = active_run.info.run_id

    worker_params = {
        'experiment_name': experiment_name,
        'mlflow_run_id': mlflow_run_id,
        'solved_thresh': solved_thresh,
        'max_steps_per_batch': steps_per_batch,
        'max_steps_per_episode': max_steps_per_episode,
        'max_steps': total_step_limit,
        'max_episodes': episode_limit,
        'metric_decay': avg_decay,
        'print_interval': log_interval,
    }

    worker_args = [
        agent_params, train_params, env_params,
        global_agent, shared_info, worker_params
    ]

    if serial:
        if n_workers == 1:
            continuous_worker_thread(0, *worker_args, lock=None)
        else:
            worker_params['max_steps'] = steps_per_batch
            worker_params['mlflow_run_id'] = None
            # Initialize the workers
            workers = []
            for i in range(n_workers):
                worker = continuous_worker_thread(i, *worker_args, lock=None)
                workers.append(worker)

            keep_training = True
            while keep_training:
                for i in range(n_workers):
                    continuous_worker_thread(
                        i, *worker_args, lock=None, worker=workers[i]
                    )
                keep_training = (
                    shared_info['solved'] == 0 and
                    shared_info['total_steps'] < total_step_limit and
                    shared_info['total_episodes'] < episode_limit
                )
    else:
        processes = []
        for i in range(n_workers):
            worker_process = mp.Process(
                target=continuous_worker_thread,
                args=[i, *worker_args],
                kwargs={'lock': lock},
            )
            worker_process.start()
            processes.append(worker_process)

        for p in processes:
            p.join()

    if use_mlflow:
        mlflow.end_run()

    solved = shared_info['solved'].item() > 0
    return global_agent, solved

