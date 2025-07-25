import torch as th

from envs.env_register import env_register
from torch_geometric.data import Data


class EpisodeRunner:
    def __init__(self, args, agent):
        self.args = args
        self.agent = agent
        self.env = env_register[args.env_name](args)
        self.t_env = 0

    def run(self, test_mode=False):
        self.agent.buffer.reset()
        self.env.reset()
        # -------------------------------------------------
        task_completion_time = 0
        failure_task_number = 0
        drop_task_number = 0
        finish_task_number = 0
        success_finish_task_number = 0
        episode_return = 0
        max_hop_dict = {}
        # -------------------------------------------------
        terminated = False
        while not terminated:
            obs = th.as_tensor(self.env.get_obs(), dtype=th.float)

            avail_actions = th.as_tensor(self.env.get_avail_actions(), dtype=th.int)
            if self.args.algo_type == 'rl':
                actions = self.agent.select_actions(obs.unsqueeze(0), avail_actions.unsqueeze(0), test_mode)
            else:
                actions = self.agent.select_actions([self.env])
            cpu_actions = actions.to("cpu").numpy()
            reward, terminated, info = self.env.step(cpu_actions[0])
            # -------------------------------------------------
            episode_return += reward
            task_completion_time += info["task_completion_time"]
            failure_task_number += info["failure_task_number"]
            drop_task_number += info["drop_task_number"]
            finish_task_number += info["finish_task_number"]
            success_finish_task_number += info["success_finish_task_number"]
            for max_hop, count in info["max_hop_dict"].items():
                if max_hop in max_hop_dict:
                    max_hop_dict[max_hop] += count
                else:
                    max_hop_dict[max_hop] = count
            # -------------------------------------------------
            rewards = th.as_tensor(reward).unsqueeze(0).unsqueeze(0).repeat(self.args.n_agents, 1)
            masks = th.as_tensor(1 - int(terminated)).unsqueeze(0).repeat(self.args.n_agents, 1)
            if not test_mode:
                self.t_env += 1
            next_obs = th.as_tensor(self.env.get_obs(), dtype=th.float)

            if not test_mode:
                self.agent.buffer.insert(obs.unsqueeze(0), avail_actions.unsqueeze(0), actions.unsqueeze(0).unsqueeze(-1), rewards.unsqueeze(0), masks.unsqueeze(0), next_obs.unsqueeze(0))
        episode_info = {}
        episode_info["episode_return"] = episode_return
        episode_info["success_rate"] = success_finish_task_number / finish_task_number
        episode_info["drop_rate"] = drop_task_number / finish_task_number
        episode_info["failure_rate"] = failure_task_number / finish_task_number
        episode_info["task_completion_time"] = task_completion_time / (success_finish_task_number + drop_task_number)
        episode_info["max_hop_dict"] = max_hop_dict
        if not test_mode:
            self.agent.train()
        return episode_info

    def close_env(self):
        self.env.close()
