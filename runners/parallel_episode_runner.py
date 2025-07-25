import copy
from multiprocessing import Pipe, Process

import torch as th

from envs.env_register import env_register


class ParallelRunner:
    def __init__(self, args, agent):
        self.args = args
        self.agent = agent

        self.parent_conns = []
        self.worker_conns = []
        self.process = []
        for i in range(self.args.n_threads):
            env_args = copy.deepcopy(args)
            env_args.task_seed += i
            env_args.failure_seed += i
            env = env_register[args.env_name](env_args)
            parent_conn, worker_conn = Pipe()
            ps = Process(target=env_worker, args=(worker_conn, env))
            ps.daemon = True
            ps.start()
            self.parent_conns.append(parent_conn)
            self.worker_conns.append(worker_conn)
            self.process.append(ps)

        self.t_env = 0

    def run(self, test_mode=False):
        self.agent.buffer.reset()
        n_obs = th.zeros(self.args.n_threads, self.args.n_agents, self.args.obs_shape, dtype=th.float)
        n_avail_actions = th.zeros(self.args.n_threads, self.args.n_agents, self.args.n_actions, dtype=th.int)
        n_rewards = th.zeros(self.args.n_threads, self.args.n_agents, 1, dtype=th.float)
        n_masks = th.zeros(self.args.n_threads, self.args.n_agents, 1, dtype=th.float)
        n_next_obs = th.zeros(self.args.n_threads, self.args.n_agents, self.args.obs_shape, dtype=th.float)

        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
        terminated = [False for _ in range(self.args.n_threads)]
        # -------------------------------------------------
        episode_return = 0
        task_completion_time = 0
        failure_task_number = 0
        drop_task_number = 0
        finish_task_number = 0
        success_finish_task_number = 0
        # -------------------------------------------------
        while True:
            if all(terminated):
                break
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    parent_conn.send(("get_obs", None))
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    n_obs[idx] = th.as_tensor(data["obs"])

            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    parent_conn.send(("get_avail_actions", None))
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    n_avail_actions[idx] = th.as_tensor(data["avail_actions"])

            # adapt greedy algorithms
            if self.args.algo_type == 'rl':
                actions = self.agent.select_actions(n_obs, n_avail_actions, test_mode)
            else:
                # ! 比单进程还要慢, 不建议使用
                envs = []
                for idx, parent_conn in enumerate(self.parent_conns):
                    if not terminated[idx]:
                        parent_conn.send(("get_env", None))
                for idx, parent_conn in enumerate(self.parent_conns):
                    if not terminated[idx]:
                        data = parent_conn.recv()
                        envs.append(data["env"])
                actions = self.agent.select_actions(envs)
            cpu_actions = actions.to("cpu").numpy()
            n_actions = actions.unsqueeze(-1)

            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    parent_conn.send(("step", cpu_actions[idx]))
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # -------------------------------------------------
                    episode_return += data["reward"]
                    task_completion_time += data["info"]["task_completion_time"]
                    failure_task_number += data["info"]["failure_task_number"]
                    drop_task_number += data["info"]["drop_task_number"]
                    finish_task_number += data["info"]["finish_task_number"]
                    success_finish_task_number += data["info"]["success_finish_task_number"]
                    # -------------------------------------------------
                    n_rewards[idx] = th.as_tensor(data["reward"]).unsqueeze(0).unsqueeze(0).repeat(self.args.n_agents, 1)
                    n_masks[idx] = th.as_tensor(1 - int(data["terminated"])).unsqueeze(0).unsqueeze(0).repeat(self.args.n_agents, 1)
                    if not test_mode:
                        self.t_env += 1
                    terminated[idx] = data["terminated"]

            for parent_conn in self.parent_conns:
                parent_conn.send(("get_obs", None))
            for idx, parent_conn in enumerate(self.parent_conns):
                data = parent_conn.recv()
                n_next_obs[idx] = th.as_tensor(data["obs"])
            if not test_mode:
                self.agent.buffer.insert(n_obs, n_avail_actions, n_actions, n_rewards, n_masks, n_next_obs)

        episode_info = {}
        episode_info["episode_return"] = episode_return / self.args.n_threads
        episode_info["success_rate"] = success_finish_task_number / finish_task_number
        episode_info["drop_rate"] = drop_task_number / finish_task_number
        episode_info["failure_rate"] = failure_task_number / finish_task_number
        episode_info["task_completion_time"] = task_completion_time / (success_finish_task_number + drop_task_number)
        if not test_mode:
            self.agent.train()
        return episode_info

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))


def env_worker(remote, env):
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            env.reset()
        elif cmd == 'step':
            actions = data
            reward, terminated, env_info = env.step(actions)
            remote.send({
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "get_obs":
            remote.send({
                "obs": env.get_obs()
            })
        elif cmd == 'get_avail_actions':
            remote.send({
                "avail_actions": env.get_avail_actions()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_env":
            remote.send({
                "env": env
            })
        else:
            raise NotImplementedError
