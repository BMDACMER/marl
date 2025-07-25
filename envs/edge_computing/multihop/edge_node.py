import math
from collections import deque
import networkx as nx
import numpy as np

from envs.edge_computing.multihop.task import Task
from envs.edge_computing.multihop.network_graph import generate_graph, generate_graph2
import warnings

warnings.filterwarnings("ignore")

K = 1024
M = 1024 * 1024
G = 1024 * 1024 * 1024
Byte = 8


class EdgeNode:
    graph = None
    g = None

    def __init__(self, args, random_state, i):
        self.args = args
        self.id = i
        # todo
        self.edge_node_random = random_state
        # self.edge_node_random = np.random.RandomState(args.edge_node_seed + i)
        self.task_random = np.random.RandomState(args.task_seed + i)
        self.failure_random = np.random.RandomState(args.failure_seed + i)
        self.edge_nodes = []  # For implementing communication between edge nodes in simulation environment
        self.cpu_core_num = self.edge_node_random.choice(args.cpu_core_list)
        self.k = self.args.k  # Maximum number of tasks that can be executed in parallel
        self.cpu_capacity = self.cpu_core_num * G * args.single_core_cpu_capacity * args.beta  # Adjust beta to simulate redundancy operations
        self.task_probability = self.edge_node_random.uniform(args.task_probability_min,
                                                              args.task_probability_max)  # Task arrival follows Bernoulli distribution

        # First get the network topology graph
        self.graph = self.get_graph(args)   # Get synthetic network topology
        # self.graph = self.get_graph2(args)  # Get real network topology
        
        # Initialize transmission rate and failure rate lists based on actual network topology node count
        actual_node_num = self.graph.num_nodes
        self.transmission_rates = [0 for _ in range(actual_node_num)]
        self.transmission_failure_rates = [0 for _ in range(actual_node_num)]
        
        self.matrix_list = self.graph.get_all_edges_3()    # Get topology edges
        adj_nodes = self.graph.get_adj_tuple(i)  # Return triple (node_id, bandwidth, failure_rate)
        for adj_node in adj_nodes:
            # Ensure index is within valid range
            if adj_node[0] < len(self.transmission_rates):
                self.transmission_rates[adj_node[0]] = adj_node[1]
                self.transmission_failure_rates[adj_node[0]] = adj_node[2]

        self.actions = self.graph.get_adj_node(i)  # Get adjacent nodes (single hop)
        # print(self.actions)
        self.actions.append(self.id)  # Current node can also decide whether to offload
        self.execution_failure_rate = self.edge_node_random.uniform(args.execution_failure_rate_min,
                                                                    args.execution_failure_rate_max)
        # Ensure current node index is within valid range
        if i < len(self.transmission_failure_rates):
            self.transmission_failure_rates[i] = 0
        self.execution_queue = deque()  # Execution queue
        self.receiving_queues = [deque() for _ in range(actual_node_num)]  # OFDMA N-1 queue  For coding convenience, initialized as N
        self.execution_queue_len = self.cpu_core_num / args.cpu_core_list[0]  # len(execution_queue) \in [4/4,32/4]=[1,8]

        # Add buffer queue--(responsible for receiving new tasks and tasks forwarded by nodes)
        self.buffer_queue = deque()  # Used to store tasks
        self.executing_queue = deque()  # Statistics of tasks running on the node
        """
        adaptive queue length
        """
        self.new_task = None
        # Whenever adding attributes, need to modify obs_shape
        self.obs_shape = 9 + actual_node_num * 0  # Number of for loops in obs  8--> 10  Added task.hop and len(buffer_queue)
        """step information"""
        self.task_completion_time = 0
        self.failure_task_number = 0
        self.drop_task_number = 0
        self.finish_task_number = 0
        self.success_finish_task_number = 0
        self.reward = 0
        self.max_hop_dict = {}  # Convenient for statistics of hop count required when tasks are completed
        self.max_execution_len = 0
        self.max_executing_len = 0
        self.task_bak = None  # Store redundant tasks

    @classmethod
    def get_graph(self, args):
        if self.graph is None:
            self.graph, self.g = generate_graph(args)  # Synthetic topology graph
        return self.graph

    @classmethod
    def get_graph2(self, args):
        if self.graph is None:
            self.graph = generate_graph2(args)  # generate real world network topology
        return self.graph

    def get_waiting_time(self):
        """Total waiting time of all tasks in current execution queue"""
        waiting_time = 0
        for task in self.execution_queue:
            waiting_time += task.execute_time - task.current_execute_time
        return waiting_time

    def get_edges(self):
        return self.matrix_list

    def get_observation(self):
        observation = []
        # edge node execution failure rate
        observation.append(self.execution_failure_rate / self.args.execution_failure_rate_max)
        # edge node CPU capacity
        observation.append(self.cpu_core_num / self.args.cpu_core_list[-1])
        # task arriving rate
        observation.append(self.task_probability / self.args.task_probability_max)
        # execution queue length
        observation.append(len(self.execution_queue) / self.execution_queue_len)
        # Execution queue and currently executing queue
        # observation.append(len(self.executing_queue) / (self.max_executing_len + 1))    # Length of currently executing queue (adding these two items are noise)
        # observation.append(len(self.execution_queue) / (self.max_execution_len + 1))    # Execution queue length (adding these two items are noise)
        # waiting time
        observation.append(self.get_waiting_time() / self.args.deadline)
        # task information
        if self.new_task:
            observation.append(self.new_task.task_size / (self.args.task_size_max * K * Byte))
            observation.append(
                self.new_task.task_cpu_cycle / (self.args.task_complexity_max * self.args.task_size_max * K * Byte))
            # 1
            observation.append(self.new_task.task_deadline / self.args.deadline)
            # observation.append(self.new_task.hop / self.args.max_hops)
            # Temporarily no limit on maximum hop count
            observation.append(self.new_task.hop)
        else:
            observation.extend([-1, -1, -1, 0])
        return observation

    def get_observation_llm(self):
        observation = {}
        observation['node_id'] = self.id
        observation['cpu_capacity'] = self.cpu_core_num
        observation['cpu_utilization'] = round(len(self.execution_queue) / self.execution_queue_len, 4)
        observation['execution_failure_rate'] = round(self.execution_failure_rate, 4)
        observation['waiting_time'] = round(self.get_waiting_time(), 4)
        # task information
        if self.new_task:
            observation['task_size'] = self.new_task.task_size / (K * Byte)
            observation['task_cpu_cycle'] = self.new_task.task_cpu_cycle / (self.args.task_size_max * K * Byte)
        else:
            observation['task_size'] = 0
            observation['task_cpu_cycle'] = 0
        # transmission rate
        observation['transmission_rate'] = [round(v / max(self.transmission_rates), 4) for v in self.transmission_rates]
        # Current node offloading space
        observation['actions_space'] = self.actions

        return observation

    def generate_task(self):
        self.new_task = None
        is_task_arrival = self.task_random.binomial(1, self.task_probability)
        if is_task_arrival == 1:
            self.buffer_queue.append(Task(self.args, self.task_random))

        if len(self.buffer_queue) > 0:
            self.new_task = self.buffer_queue.popleft()

    def forward_task(self):
        self.new_task = self.buffer_queue.popleft()

    def reset_edge_node(self):
        self.execution_queue.clear()
        for receiving_queue in self.receiving_queues:
            receiving_queue.clear()
        self.buffer_queue.clear()
        self.new_task = None
        self.executing_queue.clear()

    def get_avail_actions(self):
        if self.new_task:
            adj_nodes = self.actions
            avail_actions = [1 if i in adj_nodes else 0 for i in range(self.args.edge_node_num + 1)]
            # Use queue length limit
            for i in range(self.args.edge_node_num):
                # Further design based on core count, customize maximum queue length for each node
                if len(self.edge_nodes[i].execution_queue) >= self.edge_nodes[
                    i].execution_queue_len * self.args.rl_queue_coeff:
                    avail_actions[i] = 0
            # If no other nodes can be offloaded to, execute locally
            if avail_actions.count(1) == 0:
                avail_actions[self.id] = 1
            return avail_actions
        else:
            # No tasks, no processing
            avail_actions = [0] * (self.args.edge_node_num + 1)
            avail_actions[self.args.edge_node_num] = 1
            return avail_actions

    def offload_task(self, action):
        """step information"""
        self.task_completion_time = 0
        self.failure_task_number = 0
        self.drop_task_number = 0
        self.finish_task_number = 0
        self.success_finish_task_number = 0
        self.reward = 0

        if action == self.args.edge_node_num or self.new_task is None:
            # No task arrival
            return
        # Offload the task
        task = self.new_task
        # All remaining tasks in buffer_queue must wait for the current task to complete, so add mini_time_slot uniformly
        for i in range(0, len(self.buffer_queue)):
            waiting_task = self.buffer_queue[i]
            waiting_task.buffer_waiting_time += self.args.mini_time_slot
        if action == self.id:
            task.transmission_time = 0
            # task.execute_time = self.task_execution_time
            task.execute_time = task.task_cpu_cycle / self.cpu_capacity
            task.execution_failure_rate = self.execution_failure_rate
            task.execution_reliability = math.exp(- task.execution_failure_rate * task.execute_time)
            task.transmission_failure_rate = 0
            task.transmission_reliability = 1
            self.execution_queue.append(task)
        else:
            task.transmission_time = task.task_size / self.transmission_rates[action]
            # task.execute_time = task.task_cpu_cycle / self.edge_nodes[action].cpu_capacity
            # task.execution_failure_rate = self.edge_nodes[action].execution_failure_rate
            # task.execution_reliability = math.exp(- task.execution_failure_rate * task.execute_time)
            task.transmission_failure_rate = self.transmission_failure_rates[action]
            task.transmission_reliability = math.exp(- task.transmission_failure_rate * task.transmission_time)
            task.hop += 1
            self.edge_nodes[action].receiving_queues[self.id].append(task)

    def execute_task(self):
        self.max_execution_len = max(self.max_execution_len, len(self.execution_queue))
        temp_len = min(len(self.execution_queue), self.k)
        for i in range(temp_len):
            if len(self.executing_queue) < self.k:
                self.executing_queue.append(self.execution_queue.popleft())

        self.max_executing_len = max(self.max_executing_len, len(self.executing_queue))
        index = 0
        index_drop = 0
        len_executing_queue = len(self.executing_queue)
        # Tasks in waiting queue only add one unit time
        for i in range(0, len(self.execution_queue)):
            waiting_task = self.execution_queue[i]
            waiting_task.execute_waiting_time += self.args.mini_time_slot
        # Simulate multiple tasks executing simultaneously
        while len(self.executing_queue) > 0 and (index + index_drop) < len_executing_queue:  # Prevent array out of bounds
            task = self.executing_queue[index]
            task.current_execute_time += self.args.mini_time_slot
            task_time = task.execute_waiting_time + task.current_execute_time + task.transmission_waiting_time + task.current_transmission_time + task.buffer_waiting_time
            # 1 Task fails during execution
            min_time_slot_execution_reliability = math.exp(-task.execution_failure_rate * self.args.mini_time_slot)
            execution_failure = self.failure_random.random() > min_time_slot_execution_reliability
            if execution_failure:
                self.reward += self.args.task_failure_penalty
                self.failure_task_number += 1
                self.finish_task_number += 1

                self.executing_queue.remove(task)  # Current index value has been deleted, at this time the value corresponding to index has moved back one position, cannot use popleft()

                index_drop += 1  # Count removed tasks
            # 2 Task execution time exceeds deadline
            elif task_time > task.task_deadline:
                self.reward += self.args.task_drop_penalty
                self.drop_task_number += 1
                self.finish_task_number += 1

                self.task_completion_time += task_time
                self.executing_queue.remove(task)  # 当前的index对应的值已删除，此时 index 对应的值已经往后挪动一位

                index_drop += 1  # 统计 移除的任务
            # 3 任务在截止时间内完成
            elif task.current_execute_time >= task.execute_time:
                self.reward += self.args.task_success_reward
                self.reward += self.args.task_hop_penalty * (task.hop - 1) if task.hop > 1 else 0

                self.success_finish_task_number += 1
                self.finish_task_number += 1
                task.is_success = True                # 这条代码仅对冗余起作用
                self.task_completion_time += task_time
                # ---------添加Task的max_hop
                if task.hop in self.max_hop_dict:
                    self.max_hop_dict[task.hop] += 1
                else:
                    self.max_hop_dict[task.hop] = 1
                # ---------------
                self.executing_queue.remove(task)

                index_drop += 1  # 统计 移除的任务
            else:
                index += 1  # 任务还在执行中

        # 检测等待队列中的任务, 如果超过截止时间, 给予惩罚, 并删除队列中的任务
        new_executed_queue = deque()
        for task in self.execution_queue:
            task_time = task.execute_waiting_time + task.current_execute_time + task.transmission_waiting_time + task.current_transmission_time + task.buffer_waiting_time
            if task_time > task.task_deadline:
                self.drop_task_number += 1
                self.finish_task_number += 1
                self.reward += self.args.task_drop_penalty
                self.task_completion_time += task_time
            else:
                new_executed_queue.append(task)
        self.execution_queue = new_executed_queue

    def receive_task(self):
        #  对remote queue 的处理  物理上有N-1个队列  OFDMA
        for received_queue in self.receiving_queues:
            if len(received_queue) > 0:
                task = received_queue[0]
                task.current_transmission_time += self.args.mini_time_slot
                for i in range(1, len(received_queue)):
                    waiting_task = received_queue[i]
                    waiting_task.transmission_waiting_time += self.args.mini_time_slot
                # 1 传输失败
                min_time_slot_transmission_reliability = math.exp(- task.transmission_failure_rate * self.args.mini_time_slot)
                transmission_failure = self.failure_random.random() > min_time_slot_transmission_reliability

                if transmission_failure:
                    self.failure_task_number += 1
                    self.finish_task_number += 1
                    self.reward += self.args.task_failure_penalty
                    # self.reward += - task.task_deadline / task.transmission_reliability
                    received_queue.popleft()
                # 2 传输成功
                elif task.current_transmission_time >= task.transmission_time:
                    self.reward += self.args.task_hop_penalty * (task.hop - 1) if task.hop > 1 else 0
                    self.buffer_queue.append(received_queue.popleft())
