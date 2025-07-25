import torch as th
from envs.env_register import env_register


class GAILRunner:

    def __init__(self, args, agent):
        self.args = args
        self.agent = agent
        self.env = env_register[args.env_name](args)
        self.t_env = 0

    def run(self, test_mode=False, t_expert=0):
        self.agent.buffer.reset()
        self.env.reset()

        task_completion_time = 0
        failure_task_number = 0
        drop_task_number = 0
        finish_task_number = 0
        success_finish_task_number = 0
        episode_return = 0
        max_hop_dict = {}

        terminated = False

        while not terminated:
            obs = th.as_tensor(self.env.get_obs(), dtype=th.float)
            avail_actions = th.as_tensor(self.env.get_avail_actions(), dtype=th.int)

            expert_actions = None
            if not test_mode and t_expert < self.args.expert_guidance_steps:
                expert_ratio = self.agent.get_expert_ratio(t_expert)
                use_expert = th.rand(1).item() < expert_ratio

                if use_expert:
                    expert_actions_np = self.agent.get_expert_actions(self.env)
                    if expert_actions_np is not None:
                        expert_actions = th.tensor(expert_actions_np, dtype=th.long, device=self.args.device)
                        cpu_actions = expert_actions.to("cpu").numpy()
                    else:
                        actions = self.agent.select_actions(obs.unsqueeze(0), avail_actions.unsqueeze(0), test_mode)
                        cpu_actions = actions.to("cpu").numpy()
                        if len(cpu_actions.shape) == 2 and cpu_actions.shape[0] == 1:
                            cpu_actions = cpu_actions[0]
                else:
                    actions = self.agent.select_actions(obs.unsqueeze(0), avail_actions.unsqueeze(0), test_mode)
                    cpu_actions = actions.to("cpu").numpy()
                    if len(cpu_actions.shape) == 2 and cpu_actions.shape[0] == 1:
                        cpu_actions = cpu_actions[0]
            else:
                actions = self.agent.select_actions(obs.unsqueeze(0), avail_actions.unsqueeze(0), test_mode)
                cpu_actions = actions.to("cpu").numpy()
                if len(cpu_actions.shape) == 2 and cpu_actions.shape[0] == 1:
                    cpu_actions = cpu_actions[0]

            reward, terminated, info = self.env.step(cpu_actions.tolist())

            episode_return += reward
            task_completion_time += info["task_completion_time"]
            failure_task_number += info["failure_task_number"]
            drop_task_number += info["drop_task_number"]
            finish_task_number += info["finish_task_number"]
            success_finish_task_number += info["success_finish_task_number"]
            
            if "max_hop" in info:
                max_hop = info["max_hop"]
                if max_hop in max_hop_dict:
                    max_hop_dict[max_hop] += 1
                else:
                    max_hop_dict[max_hop] = 1

            if expert_actions is not None:
                agent_actions = self.agent.select_actions(obs.unsqueeze(0), avail_actions.unsqueeze(0), test_mode)
                store_actions = agent_actions
            else:
                store_actions = actions

            rewards = th.as_tensor(reward).unsqueeze(0).unsqueeze(0).repeat(self.args.n_agents, 1)
            masks = th.as_tensor(1 - int(terminated)).unsqueeze(0).repeat(self.args.n_agents, 1)

            if not test_mode:
                self.t_env += 1

            next_obs = th.as_tensor(self.env.get_obs(), dtype=th.float)

            if not test_mode:
                self.agent.buffer.insert(
                    obs.unsqueeze(0),
                    avail_actions.unsqueeze(0),
                    store_actions.unsqueeze(0).unsqueeze(-1),
                    rewards.unsqueeze(0),
                    masks.unsqueeze(0),
                    next_obs.unsqueeze(0)
                )

                if expert_actions is not None:
                    self.agent.store_expert_data(obs, expert_actions)

        episode_info = {
            "episode_return": episode_return,
            "success_rate": success_finish_task_number / max(finish_task_number, 1),
            "drop_rate": drop_task_number / max(finish_task_number, 1),
            "failure_rate": failure_task_number / max(finish_task_number, 1),
            "task_completion_time": task_completion_time / max(success_finish_task_number + drop_task_number, 1),
            "max_hop_dict": max_hop_dict
        }

        if not test_mode:
            self.agent.update_performance_monitoring(episode_info["success_rate"])
            self.agent.train()

        return episode_info

    def close_env(self):
        self.env.close()
