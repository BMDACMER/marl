from typing import Dict, Any

import torch as th
from envs.env_register import env_register


class StepRunner:
    def __init__(self, args, agent):
        self.args = args
        self.agent = agent
        self.env = env_register[args.env_name](args)
        self.t_env = 0

    def run(self, test_mode=False):
        if self.args.name == 'drqn':
            self.agent.h = self.agent.drqn.init_hidden(train=False)
        self.env.reset()
        # -------------------------------------------------
        task_completion_time = 0
        failure_task_number = 0
        drop_task_number = 0
        finish_task_number = 0
        success_finish_task_number = 0
        episode_return = 0
        max_hop_dict: Dict[Any, int] = {}
        # -------------------------------------------------
        terminated = False
        while not terminated:
            # obs
            obs = th.as_tensor(self.env.get_obs(), dtype=th.float)
            # avail_actions
            avail_actions = th.as_tensor(self.env.get_avail_actions(), dtype=th.int)
            if self.args.algo_type == 'rl':
                actions = self.agent.select_actions(obs, avail_actions, test_mode)
            else:
                actions = self.agent.select_actions(self.env)
            cpu_actions = actions.to("cpu").numpy()
            actions = actions.unsqueeze(-1)
            reward, terminated, info = self.env.step(cpu_actions)
            # -------------------------------------------------
            episode_return += reward
            task_completion_time += info["task_completion_time"]
            failure_task_number += info["failure_task_number"]
            drop_task_number += info["drop_task_number"]
            finish_task_number += info["finish_task_number"]
            success_finish_task_number += info["success_finish_task_number"]
            for max_hop in info["max_hop_dict"]:
                if max_hop in max_hop_dict:
                    max_hop_dict[max_hop] += 1
                else:
                    max_hop_dict[max_hop] = 1
            # -------------------------------------------------
            # rewards
            rewards = th.as_tensor(reward, dtype=th.float).unsqueeze(0).unsqueeze(0).repeat(self.args.n_agents, 1)
            # masks
            masks = th.as_tensor(1 - int(terminated), dtype=th.float).unsqueeze(0).repeat(self.args.n_agents, 1)
            # next obs
            next_obs = th.as_tensor(self.env.get_obs(), dtype=th.float)
            if not test_mode:
                self.t_env += 1
                # buffer
                self.agent.buffer.insert(obs, avail_actions, actions, rewards, masks, next_obs)
                # train
                self.agent.train()
        episode_info = {}
        episode_info["episode_return"] = episode_return
        episode_info["success_rate"] = success_finish_task_number / finish_task_number
        episode_info["drop_rate"] = drop_task_number / finish_task_number
        episode_info["failure_rate"] = failure_task_number / finish_task_number
        episode_info["task_completion_time"] = task_completion_time / (success_finish_task_number + drop_task_number)
        episode_info["max_hop_dict"] = max_hop_dict
        return episode_info

    def run2(self, test_mode=False, train_data=None, id=0):
        if train_data is None:
            train_data = []
        if self.args.name == 'drqn':
            self.agent.h = self.agent.drqn.init_hidden(train=False)
        self.env.reset()
        # -------------------------------------------------
        task_completion_time = 0
        failure_task_number = 0
        drop_task_number = 0
        finish_task_number = 0
        success_finish_task_number = 0
        episode_return = 0
        reward = 0
        # -------------------------------------------------
        terminated = False
        while not terminated:
            # obs
            obs = th.as_tensor(self.env.get_obs(), dtype=th.float)
            # avail_actions
            avail_actions = th.as_tensor(self.env.get_avail_actions(), dtype=th.int)
            if self.args.algo_type == 'rl':
                actions = self.agent.select_actions(obs, avail_actions, test_mode)
            else:
                actions = self.agent.select_actions(self.env)
            cpu_actions = actions.to("cpu").numpy()
            cpu_a = cpu_actions.tolist()
            actions = actions.unsqueeze(-1)
            reward, terminated, info = self.env.step(cpu_actions)
            #################################################
            sf = info["success_finish_task_number"]
            fn = info["finish_task_number"]
            if reward == 0 or (fn != 0 and sf / fn == 1):
                # print(f"reward: {reward}\t success_finish_task_number:", str(sf), "\t finish_task_number: ", str(fn))
                # cpu_a = cpu_actions.tolist()  # index_id
                load_information = self.env.get_obs_llm()
                cpu_capacity = []
                cpu_utilization_rate = []
                execution_failure_rate = []
                waiting_time = []
                node_id = []
                task_size = []
                computing_res_required = []
                bandwidths = []
                actions_space = []

                for load_info in load_information:
                    cpu_capacity.append(load_info['cpu_capacity'])
                    cpu_utilization_rate.append(load_info['cou_utilization'])
                    execution_failure_rate.append(load_info['execution_failure_rate'])
                    waiting_time.append(load_info['waiting_time'])
                    node_id.append(load_info['node_id'])
                    task_size.append(load_info['task_size'])
                    computing_res_required.append(load_info['task_cpu_cycle'])
                    bandwidths.append(load_info['transmission_rate'])
                    actions_space.append(load_info['actions_space'])

                for i in range(self.args.edge_node_num):
                    prompt_template = f"""There are now {self.args.edge_node_num} heterogeneous compute resource nodes. Numbered from 0 to {self.args.edge_node_num - 1}. The CPU capacity and utilization rate of each node are respectively as follows : {cpu_capacity} and {cpu_utilization_rate}. The execution failure rate of each node  is: {execution_failure_rate}. The current node is {i}, which can offload the task to the current node or its neighbor nodes, and the decision space at this node is {actions_space[i]}. Now a task arrives at node {i}, and task size is {task_size[i]} and computing resources required is {computing_res_required[i]} and the transmission rates to each node for the task is {bandwidths[i]}. Note that a transmission rate of 0 means that two nodes are not connected. If the task size is 0, it means that there is no task at this time, and the offloading node index is {self.args.edge_node_num}. \n Based on the above information, the task offloading node index is: """
                    # 通义千问的数据集模板
                    # example = {
                    #     'id': 'identity_{}'.format(id),
                    #     'conversations': [
                    #         {
                    #             "from": "user",
                    #             "value": prompt_template,
                    #         },
                    #         {
                    #             "from": "assistant",
                    #             "value": str(cpu_a[i]),
                    #         },
                    #     ]
                    # }
                    example = {
                        "instruction": prompt_template,
                        "input": "",
                        "output": str(cpu_a[i])
                    }
                    id += 1
                    # print(example)
                    train_data.append(example)
            #################################################
            # -------------------------------------------------
            episode_return += reward
            task_completion_time += info["task_completion_time"]
            failure_task_number += info["failure_task_number"]
            drop_task_number += info["drop_task_number"]
            finish_task_number += info["finish_task_number"]
            success_finish_task_number += info["success_finish_task_number"]
            # -------------------------------------------------
            # rewards
            rewards = th.as_tensor(reward, dtype=th.float).unsqueeze(0).unsqueeze(0).repeat(self.args.n_agents, 1)
            # masks
            masks = th.as_tensor(1 - int(terminated), dtype=th.float).unsqueeze(0).repeat(self.args.n_agents, 1)
            # next obs
            next_obs = th.as_tensor(self.env.get_obs(), dtype=th.float)
            if not test_mode:
                self.t_env += 1
                # buffer
                self.agent.buffer.insert(obs, avail_actions, actions, rewards, masks, next_obs)
                # train
                self.agent.train()
        episode_info = {}
        episode_info["episode_return"] = episode_return
        episode_info["success_rate"] = success_finish_task_number / finish_task_number
        episode_info["drop_rate"] = drop_task_number / finish_task_number
        episode_info["failure_rate"] = failure_task_number / finish_task_number
        episode_info["task_completion_time"] = task_completion_time / (
                    success_finish_task_number + drop_task_number)
        return episode_info, train_data, id

    def close_env(self):
        self.env.close()
