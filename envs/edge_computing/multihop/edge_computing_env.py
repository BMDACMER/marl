import numpy as np

from envs.edge_computing.multihop.edge_node import EdgeNode
from envs.multiagentenv import MultiAgentEnv


class EdgeComputingEnv(MultiAgentEnv):
    def __init__(self, args):
        edge_node_random = np.random.RandomState(args.edge_node_seed)
        self.edge_nodes = []
        for i in range(args.edge_node_num):
            # pass the parameters.
            edge_node = EdgeNode(args, edge_node_random, i)
            self.edge_nodes.append(edge_node)

        self.current_step = 0
        for i in range(args.edge_node_num):
            self.edge_nodes[i].generate_task()
        # Pass the list of all nodes in the entire edge network to each node for convenient inter-node communication. This approach allows each node to easily access all other nodes in the edge network.
        for i in range(args.edge_node_num):
            self.edge_nodes[i].edge_nodes = self.edge_nodes

        # No user tasks, no offloading decisions, occupies the last position
        args.n_actions = args.edge_node_num + 1
        args.n_agents = args.edge_node_num
        self.args = args

        # Set obs and state sizes
        self.obs_shape = self.edge_nodes[0].obs_shape
        self.state_shape = self.obs_shape * self.args.edge_node_num

    def get_obs(self):
        """Returns all agent observations in a list"""
        agent_obs_list = []
        for i in range(self.args.edge_node_num):
            agent_obs_list.append(self.edge_nodes[i].get_observation())
        return np.array(agent_obs_list)

    def get_edges(self):
        """Returns edges of all nodes"""
        return np.array(self.edge_nodes[0].get_edges())

    def get_obs_llm(self):
        """Returns all agent observations  as llm input"""
        all_agent_obs = []
        for i in range(self.args.edge_node_num):
            all_agent_obs.append(self.edge_nodes[i].get_observation_llm())
        return all_agent_obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        obs = self.edge_nodes[agent_id].get_observation()
        return np.array(obs)

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.obs_shape

    def get_state(self):
        """This method is the same as get_obs, not used for now"""
        state = []
        for i in range(self.args.edge_node_num):
            state.extend(self.edge_nodes[i].get_observation())
        return np.array(state)

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.state_shape

    def get_avail_agent_actions(self, agent_id):
        return self.edge_nodes[agent_id].get_avail_actions()

    def get_avail_actions(self):
        result = []
        for i in range(self.args.edge_node_num):
            result.append(self.get_avail_agent_actions(i))
        return result

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # This is only suitable for a discrete 1 dimensional action space for each agent
        return self.args.edge_node_num + 1

    def step(self, actions):
        """all edge node information"""
        task_completion_time = 0
        failure_task_number = 0
        drop_task_number = 0
        finish_task_number = 0
        success_finish_task_number = 0
        reward = 0
        max_hop_dict = {}
        max_execution_len = 0
        max_buffer_len = 0
        max_executing_len = 0

        for i in range(len(actions)):
            self.edge_nodes[i].offload_task(actions[i])

        # Let the simulation more closed to real environment: execute and receive tasks simultaneously, simulate parallel operations
        for i in range(self.args.mini_time_slot_num):
            for edge_node in self.edge_nodes:
                edge_node.execute_task()
            for edge_node in self.edge_nodes:
                edge_node.receive_task()

        # Statistics of step information for all edge nodes
        for edge_node in self.edge_nodes:
            reward += edge_node.reward
            task_completion_time += edge_node.task_completion_time
            failure_task_number += edge_node.failure_task_number
            drop_task_number += edge_node.drop_task_number
            finish_task_number += edge_node.finish_task_number
            success_finish_task_number += edge_node.success_finish_task_number
            # Statistics of maximum hop count for successfully completed tasks on each node
            for max_hop, count in edge_node.max_hop_dict.items():
                if max_hop in max_hop_dict:
                    max_hop_dict[max_hop] += count
                else:
                    max_hop_dict[max_hop] = count
            max_execution_len = max(max_execution_len, edge_node.max_execution_len)
            max_buffer_len = max(max_buffer_len, len(edge_node.buffer_queue))
            max_executing_len = max(max_executing_len, edge_node.max_executing_len)

        self.current_step += 1

        for edge_node in self.edge_nodes:
            edge_node.generate_task()

        info = {
            "finish_task_number": finish_task_number,
            "success_finish_task_number": success_finish_task_number,
            "drop_task_number": drop_task_number,
            "failure_task_number": failure_task_number,
            "task_completion_time": task_completion_time,
            "max_hop_dict": max_hop_dict,
        }

        if self.current_step >= self.args.episode_limit:
            terminated = True
        else:
            terminated = False
        return reward, terminated, info

    def reset(self):
        self.current_step = 0
        for edge_node in self.edge_nodes:
            edge_node.reset_edge_node()

    def close(self):
        pass

    def get_env_info(self):
        return {
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "n_actions": self.args.edge_node_num + 1,
            "n_agents": self.args.edge_node_num,
            "episode_limit": self.args.episode_limit,
        }
