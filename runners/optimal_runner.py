import torch as th
import numpy as np
import json
from envs.env_register import env_register


class OptimalRunner:

    def __init__(self, args, agent):

        self.args = args
        self.agent = agent
        self.env = env_register[args.env_name](args)
        
        # LLM训练数据收集
        self.training_data = []
    
    def run(self, test_mode=True):
        self.env.reset()
        
        optimal_actions, optimal_reward = self.agent.run_optimal_search(self.env)

        if optimal_actions is not None:
            episode_info = self._verify_and_collect_training_data(optimal_actions, optimal_reward)
            return episode_info
        else:
            return self._create_empty_episode_info()
    
    def _verify_and_collect_training_data(self, optimal_actions, expected_reward):
        self.env.reset()
        total_reward = 0
        episode_info = {
            'task_completion_time': 0,
            'failure_task_number': 0,
            'drop_task_number': 0,
            'finish_task_number': 0,
            'success_finish_task_number': 0,
            'max_hop_dict': {}
        }
        
        step_count = 0
        
        for step, actions in enumerate(optimal_actions):
            self._collect_step_training_data(step, actions)
            
            reward, terminated, info = self.env.step(actions)
            total_reward += reward
            step_count += 1
            
            episode_info['task_completion_time'] += info.get('task_completion_time', 0)
            episode_info['failure_task_number'] += info.get('failure_task_number', 0)
            episode_info['drop_task_number'] += info.get('drop_task_number', 0)
            episode_info['finish_task_number'] += info.get('finish_task_number', 0)
            episode_info['success_finish_task_number'] += info.get('success_finish_task_number', 0)
            
            for hop, count in info.get('max_hop_dict', {}).items():
                if hop in episode_info['max_hop_dict']:
                    episode_info['max_hop_dict'][hop] += count
                else:
                    episode_info['max_hop_dict'][hop] = count
            
            if terminated:
                break
        
        self._calculate_episode_stats(episode_info)
        episode_info['episode_return'] = total_reward
        episode_info['step_count'] = step_count
        
        reward_diff = abs(total_reward - expected_reward)
        if reward_diff > 1e-6:
            print(f"Warning: Reward validation failed! Expectations:  {expected_reward:.6f}, Actual: {total_reward:.6f}")
        else:
            print(f"✓ Reward verification passed: {total_reward:.6f}")
        
        # 打印统计信息
        print(f"✓ Success rate: {episode_info['success_rate']:.4f}")
        print(f"✓ Fail rate: {episode_info['failure_rate']:.4f}")
        print(f"✓ Drop rate: {episode_info['drop_rate']:.4f}")
        print(f"✓ Total steps: {step_count}")
        print(f"✓ The number of training samples is collected: {len(self.training_data)}")
        
        return episode_info
    
    def _collect_step_training_data(self, step, optimal_actions):
        try:
            load_information = self.env.get_obs_llm()
            
            cpu_capacity = []
            cpu_utilization_rate = []
            execution_failure_rate = []
            task_size = []
            computing_res_required = []
            bandwidths = []
            actions_space = []
            
            for load_info in load_information:
                cpu_capacity.append(self._convert_to_native_type(load_info['cpu_capacity']))
                cpu_utilization_rate.append(self._convert_to_native_type(load_info['cpu_utilization']))
                execution_failure_rate.append(self._convert_to_native_type(load_info['execution_failure_rate']))
                task_size.append(self._convert_to_native_type(load_info['task_size']))
                computing_res_required.append(self._convert_to_native_type(load_info['task_cpu_cycle']))
                bandwidths.append([self._convert_to_native_type(x) for x in load_info['transmission_rate']])
                actions_space.append([self._convert_to_native_type(x) for x in load_info['actions_space']])
            
            for i in range(self.args.edge_node_num):
                if task_size[i] > 0:
                    prompt_template = self._generate_prompt_template(
                        i, cpu_capacity, cpu_utilization_rate, execution_failure_rate,
                        task_size, computing_res_required, bandwidths, actions_space
                    )
                    
                    example = {
                        "instruction": prompt_template,
                        "input": "You are a senior expert in the field of edge computing and need to make optimal task scheduling decisions based on the information provided by users.",
                        "output": str(optimal_actions[i])
                    }
                    
                    self.training_data.append(example)
                    
        except Exception as e:
            print(f"Error while collecting training data: {e}")
    
    def _convert_to_native_type(self, value):
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return value
    
    def _generate_prompt_template(self, node_id, cpu_capacity, cpu_utilization_rate, 
                                execution_failure_rate, task_size, computing_res_required, 
                                bandwidths, actions_space):
        prompt = (
            f"There are now {self.args.edge_node_num} heterogeneous compute resource nodes. "
            f"Numbered from 0 to {self.args.edge_node_num - 1}. "
            f"The CPU capacity and utilization rate of each node are respectively as follows : "
            f"{cpu_capacity} and {cpu_utilization_rate}. "
            f"The execution failure rate of each node is: {execution_failure_rate}. "
            f"The current node is {node_id}, which can offload the task to the current node "
            f"or its neighbor nodes, and the decision space at this moment is {actions_space[node_id]}. "
            f"Now a task arrives at node {node_id}, and task size is {task_size[node_id]} "
            f"and computing resources required is {computing_res_required[node_id]} "
            f"and the transmission rates to each node for the task is {bandwidths[node_id]}. "
            f"Note that a transmission rate of 0 means that two nodes are not connected. "
            f"If the task size is 0, it means that there is no task at this time, "
            f"and the offloading node index is {self.args.edge_node_num}. "
            f"\n Based on the above information, the task offloading node index is: "
        )
        return prompt
    
    def _calculate_episode_stats(self, episode_info):
        if episode_info['finish_task_number'] > 0:
            episode_info['success_rate'] = episode_info['success_finish_task_number'] / episode_info['finish_task_number']
            episode_info['failure_rate'] = episode_info['failure_task_number'] / episode_info['finish_task_number']
            episode_info['drop_rate'] = episode_info['drop_task_number'] / episode_info['finish_task_number']
            
            completed_tasks = episode_info['success_finish_task_number'] + episode_info['drop_task_number']
            if completed_tasks > 0:
                episode_info['avg_completion_time'] = episode_info['task_completion_time'] / completed_tasks
            else:
                episode_info['avg_completion_time'] = 0
        else:
            episode_info['success_rate'] = 0
            episode_info['failure_rate'] = 0
            episode_info['drop_rate'] = 0
            episode_info['avg_completion_time'] = 0
    
    def _create_empty_episode_info(self):
        return {
            'episode_return': 0,
            'success_rate': 0,
            'failure_rate': 0,
            'drop_rate': 0,
            'avg_completion_time': 0,
            'step_count': 0,
            'task_completion_time': 0,
            'failure_task_number': 0,
            'drop_task_number': 0,
            'finish_task_number': 0,
            'success_finish_task_number': 0,
            'max_hop_dict': {}
        }
    
    def get_training_data(self):
        return self.training_data
    
    def clear_training_data(self):
        self.training_data = []