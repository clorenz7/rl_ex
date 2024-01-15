from contextlib import nullcontext
import datetime
import math
import torch.multiprocessing as mp
import os
import time

import mlflow
import torch

import cor_rl.agents
from cor_rl import environments
from cor_rl import utils
from cor_rl.agents.a2c import (
    InteractionResult,
    AdvantageActorCriticAgent
)

MLFLOW_URI = "http://127.0.0.1:8888"
mlflow.set_tracking_uri(uri=MLFLOW_URI)


def interact(env, agent, t_max=5, state=None, output_frames=False, lock=None):
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
        action_idx, value_est, entropy, log_prob = agent.select_action(
            state, lock=lock
        )
        state, reward, terminated, _, _ = env.step(action_idx)

        results.rewards.append(reward)
        results.values.append(value_est)
        results.log_probs.append(log_prob)
        results.entropies.append(entropy)

        if output_frames:
            frame_buffer.append(env.render())
        t += 1

    if terminated:
        # Having this be a float and not tensor is important for stability
        value_est = 0.0
        state = None
    # elif state is None:
    #     # Lost a life: episode restart. Take a no-op so state is not None
    #     # so that it doesn't trigger an environment reset
    #     for _ in range(1):
    #         state, reward, terminated_no_op, _, _ = env.step(0)
    #         # Just in case there were points scored after death
    #         results.rewards[-1] += reward

    #     # Having this be a float and not tensor is important for stability
    #     value_est = 0.0
    else:
        # Get an estimate of the value of the final state
        with torch.no_grad():
            action_idx, value_est, _, _ = agent.select_action(state, lock=lock)
        value_est = value_est.item()

    results.values.append(value_est)

    return results, state, terminated, frame_buffer


class Worker:

    def __init__(self, task_id, agent_params, train_params, env_params,
                 shared_agent: AdvantageActorCriticAgent, shared_opt=None,
                 eval_mode=False, render=False, lock=None, steps_per_epoch=1e6):
        self.eval_mode = eval_mode
        self.task_id = task_id
        self.lock = lock or nullcontext()
        self.steps_per_epoch = steps_per_epoch
        agent_params = dict(agent_params)
        env_params = dict(env_params)
        if eval_mode:
            # Turn off reward clipping
            env_params.pop('reward_clip', None)
            agent_params['reward_clip'] = None

        if render and task_id == 0:
            env_params['render_mode'] = 'human'
        else:
            env_params['render_mode'] = 'rgb_array'

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

        self.agent = cor_rl.agents.factory(agent_params, train_params)
        self.shared_agent = shared_agent
        self.shared_opt = shared_opt

        self.frames = []
        self.reset_metrics()

    def save_on_interval(self, shared_info, worker_params={}):
        save_interval = worker_params.get('epoch_save_interval', 2)
        out_dir = worker_params.get('out_dir') or utils.DEFAULT_DIR
        experiment_name = worker_params.get('experiment_name', 'unk')
        now = datetime.datetime.now().strftime("%Y_%b_%d_H%H_%M")
        run_name = worker_params.get('run_name') or now
        max_steps = worker_params.get('max_steps') or 200e6
        max_episodes = worker_params.get('max_episodes') or 1e9
        stay_alive = True
        last_save = 0
        timeout = worker_params.get('timeout', 3600 * 24 * 10)

        while stay_alive:
            self.start_time = time.time()
            epochs = shared_info['total_steps'] / self.steps_per_epoch
            if (epochs - last_save) >= save_interval:
                with self.lock:
                    self.agent.set_parameters(
                        self.shared_agent.get_parameters()
                    )
                last_save = int(
                    math.floor(epochs / save_interval) * save_interval
                )
                # Do the save
                file_name = os.path.join(
                    out_dir, f'{experiment_name}_{run_name}_epoch{last_save}.pt'
                )
                self.agent.checkpoint(file_name)
                print(f'Saved Agent to {file_name}!')
                # Wait 5 minutes
                time.sleep(5 * 60)
            else:
                # Wait 1 minute and try again
                time.sleep(60)
            stay_alive = (
                shared_info['total_episodes'] < max_episodes and
                shared_info['total_steps'] < max_steps and
                (time.time() - self.start_time) < timeout
            )

    def reset_metrics(self):
        self.metrics = dict(
            ep_score=0, ep_loss=0,
            ep_grad_norm=0.0,
            max_loss=float('-inf'),
            max_grad_norm=float('-inf'),
            ep_steps=0, ep_batches=0
        )

    def setup_logging(self, worker_params):
        self.mlflow_run_id = worker_params.get('mlflow_run_id')
        self.use_mlflow = self.mlflow_run_id is not None
        self.metric_log_interval = worker_params.get('metric_log_interval', 4)
        self.print_interval = worker_params.get('print_interval', 10)

        if self.use_mlflow:
            mlflow.set_experiment(worker_params['experiment_name'])
            mlflow.start_run(self.mlflow_run_id, nested=True)
            if self.task_id == 0:
                params = self.agent.params()
                params['repeat_action_probability'] = self.env.spec.kwargs.get('repeat_action_probability', -0.01)
                params['seed'] = self.seed
                mlflow.log_params(params)

        self.start_time = time.time()

    def teardown_logging(self):
        pass

    def log_metrics(self, episode_num, shared_info):
        avg_score = shared_info['avg_score'].item()
        if (episode_num % self.print_interval) == 0:
            elap_time = (time.time() - self.start_time) / 60.0
            epoch = shared_info['total_steps'].item() / self.steps_per_epoch
            print(
                f"Epoch: {epoch:0.2f}\t"
                f"Episode {episode_num:.0f}\t"
                f"Avg Score: {avg_score:0.2f} \t"
                f"Time: {elap_time:0.2f}min\t"
            )
        if episode_num % self.metric_log_interval == 0:
            if self.use_mlflow:
                metrics = {
                    'running_avg_score': avg_score,
                    'running_avg_loss': shared_info['avg_loss'].item(),
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
        keep_training, solved = True, False
        metric_decay = worker_params.get('metric_decay', 0.95)
        solved_thresh = worker_params.get('solved_thresh') or float('inf')
        max_steps_per_batch = worker_params.get('max_steps_per_batch', 5)
        max_steps = worker_params.get('max_steps') or 200e6
        max_episodes = worker_params.get('max_episodes') or 1e9
        save_frames = worker_params.get('save_frames', False)
        single_batch = worker_params.get('single_batch', False)

        if not single_batch:
            self.reset_metrics()
        self.setup_logging(worker_params)

        while keep_training:
            self.agent.set_parameters(self.shared_agent.get_parameters())

            self.agent.zero_grad(set_to_none=True)

            results, state, terminated, frames = interact(
                self.env, self.agent, t_max=max_steps_per_batch,
                state=self.state, output_frames=save_frames,
            )
            if save_frames:
                self.frames.extend(frames)
            self.state = state
            loss, norm_val = self.agent.calc_loss_and_backprop(results)
            if shared_info['solved'].item() == 0:
                with self.lock:
                    self.shared_opt.zero_grad(set_to_none=True)
                    self.agent.sync_grads(self.shared_agent)
                    self.shared_opt.step()
            else:
                break

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
            if single_batch:
                break

        self.teardown_logging()


def continuous_worker_thread(task_id, agent_params, train_params, env_params,
                             shared_agent, shared_info, worker_params, lock=None,
                             worker=None, render=False, save_mode=False, setup_only=False):
    if worker is None:
        worker = Worker(
            task_id, agent_params, train_params, env_params,
            shared_agent=shared_agent, shared_opt=shared_agent.optimizer,
            eval_mode=False, render=render, lock=lock
        )

    if setup_only:
        pass
    elif save_mode:
        worker.save_on_interval(shared_info, worker_params)
    else:
        worker.continuously_train(shared_info, worker_params)

    return worker


def train_loop_continuous(n_workers, agent_params, train_params, env_params,
                          steps_per_batch=5, total_step_limit=10000,
                          episode_limit=None, max_steps_per_episode=10000,
                          solved_thresh=None, log_interval=1e9, seed=8888,
                          avg_decay=0.95, debug=False, out_dir=None,
                          eval_interval=None, accumulate_grads=False,
                          experiment_name=None, load_file=None, save_interval=None,
                          use_mlflow=False, serial=False, shared_mode=True,
                          render=False, use_lock=True, repro_mode=False,
                          metric_log_interval=4, save_gif='', run_name=None):

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
        active_run = mlflow.start_run(run_name=run_name)
        mlflow.log_param('n_workers', n_workers)
        mlflow_run_id = active_run.info.run_id
        if not run_name:
            run_name = active_run.info.run_name

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
        'epoch_save_interval': save_interval,
        'out_dir': out_dir,
        'metric_log_interval': metric_log_interval,
        'save_frames': bool(save_gif),
        'run_name': run_name,
    }

    worker_args = [
        agent_params, train_params, env_params,
        global_agent, shared_info, worker_params
    ]

    if serial:
        if n_workers == 1:
            worker = continuous_worker_thread(0, *worker_args, lock=None, render=render)
            if save_gif:
                utils.write_gif(worker.frames, save_gif, fps=30)

        else:
            worker_params['single_batch'] = True
            worker_params['mlflow_run_id'] = None
            # Initialize the workers
            workers = []
            for i in range(n_workers):
                worker = continuous_worker_thread(i, *worker_args, lock=lock, setup_only=True)
                workers.append(worker)

            # Set seed to have same action selection in serial (debugging) mode
            if seed and not repro_mode:
                torch.manual_seed(seed)

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
                    if not keep_training:
                        break
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

        if save_interval is None:
            print("Will not be saving agent checkpoints!")
        else:
            # A thread that saves every epoch interval
            save_process = mp.Process(
                target=continuous_worker_thread,
                args=[i, *worker_args],
                kwargs={'save_mode': True, 'lock': lock},
            )
            processes.append(save_process)
            save_process.start()

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            import ipdb; ipdb.set_trace()
        finally:
            if use_mlflow:
                mlflow.end_run()

    solved = shared_info['solved'].item() > 0
    return global_agent, solved

