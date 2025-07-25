import copy
import math
from buffer.episode_buffer import EpisodeBuffer
from buffer.optimal_buffer import OptimalBuffer
import time
import torch as th
import numpy as np
from envs.env_register import env_register


class OptimalAgent:
    """
    Centralized optimal scheduling agent based on parameter optimization

    Core strategy: Optimize algorithm performance through systematic parameter tuning
    """

    def __init__(self, args):
        self.args = args
        self.device = getattr(args, 'device', 'cuda' if th.cuda.is_available() else 'cpu')
        self.buffer = EpisodeBuffer(args)
        self.optimal_buffer = OptimalBuffer(args)

        # Search related parameters
        self.max_search_depth = getattr(args, 'episode_limit', 100)
        self.best_solution = None
        self.best_reward = float('-inf')
        self.search_count = 0

        # Key parameter tuning: optimization weight configuration
        self.optimization_weights = {
            'reliability_weight': 0.6,  # Reliability weight (increased)
            'deadline_weight': 0.25,  # Deadline weight
            'load_balance_weight': 0.1,  # Load balance weight (decreased)
            'efficiency_weight': 0.05  # Efficiency weight (decreased)
        }

        # Key parameter tuning: threshold parameters
        self.threshold_params = {
            'max_queue_utilization': 0.7,  # Maximum queue utilization (decreased)
            'min_time_margin': 0.4,  # Minimum time margin (increased)
            'max_failure_rate': 0.15,  # Maximum acceptable failure rate (decreased)
            'preferred_local_threshold': 0.8,  # Local execution preference threshold
            'reliability_threshold': 0.85  # Reliability threshold (increased)
        }

        # Key parameter tuning: strategy parameters
        self.strategy_params = {
            'local_preference_bonus': 0.3,  # Local execution bonus (increased)
            'queue_penalty_factor': 0.2,  # Queue penalty factor (increased)
            'failure_penalty_factor': 5.0,  # Failure penalty factor (significantly increased)
            'time_pressure_factor': 3.0,  # Time pressure factor (increased)
            'conservative_factor': 1.5  # Conservative factor (newly added)
        }

        # Store detailed information of the optimal episode
        self.optimal_episode_data = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'avail_actions': [],
            'next_obs': [],
            'masks': [],
            'total_reward': 0,
            'success_rate': 0,
            'episode_info': {}
        }

    def run_optimal_search(self, env):
        """
        Execute optimal search with parameter optimization
        """
        print(f"Starting parameter optimization search, maximum search depth: {self.max_search_depth}")

        start_time = time.time()

        # Reset search state
        self.best_solution = None
        self.best_reward = float('-inf')
        self.search_count = 0

        try:
            actions_sequence1, reward1 = self._conservative_optimized_search(env)

            if reward1 > self.best_reward:
                self.best_reward = reward1
                self.best_solution = actions_sequence1.copy()

        except Exception as e:
            print(f"Parameter optimization search failed: {e}")
            # Fallback strategy
            actions_sequence, total_reward = self._fallback_search(env)
            self.best_reward = total_reward
            self.best_solution = actions_sequence.copy()
            print(f"  Fallback strategy completed! Reward: {self.best_reward:.4f}")

        end_time = time.time()
        print(f"\nSearch completed! Best reward: {self.best_reward:.4f}, Total time: {end_time - start_time:.2f}s")

        return self.best_solution, self.best_reward

    def get_current_optimal_action(self, env):
        """
        Get the optimal action for the current environment state (new method)
        
        Args:
            env: Current environment state
            
        Returns:
            numpy.ndarray: Optimal action for current state
        """
        try:
            # Use conservative action selection strategy to calculate current optimal action
            optimal_actions = self._conservative_action_selection(env, conservativeness=1.0)
            return np.array(optimal_actions)
            
        except Exception as e:
            print(f"Failed to get current optimal action: {e}")
            # Fallback strategy
            return self._get_fallback_actions(env)

    def _get_fallback_actions(self, env):
        """
        Get fallback actions
        
        Args:
            env: Environment instance
            
        Returns:
            numpy.ndarray: Fallback actions
        """
        try:
            optimal_actions = self._simple_greedy_selection(env)
            return np.array(optimal_actions)
        except Exception as e:
            print(f"Fallback action selection failed: {e}")
            # Final fallback: return agent IDs
            return np.arange(self.args.n_agents)

    def _conservative_optimized_search(self, env):
        """
        Conservative optimization strategy - extremely emphasizes reliability and time margin
        """
        env_copy = copy.deepcopy(env)
        env_copy.reset()

        actions_sequence = []
        total_reward = 0

        # Dynamically adjust conservativeness
        for step in range(self.max_search_depth):
            # As time progresses, gradually reduce conservativeness
            conservativeness = 1.0 - (step / self.max_search_depth) * 0.3
            optimal_actions = self._conservative_action_selection(env_copy, conservativeness)
            actions_sequence.append(optimal_actions)

            reward, terminated, info = env_copy.step(optimal_actions)
            total_reward += reward
            self.search_count += 1

            if terminated:
                break

        return actions_sequence, total_reward

    def _conservative_action_selection(self, env, conservativeness=1.0):
        """
        Conservative action selection - optimized version
        """
        avail_actions = env.get_avail_actions()
        n_agents = len(avail_actions)
        optimal_actions = []

        for agent_id in range(n_agents):
            available_indices = [j for j, available in enumerate(avail_actions[agent_id]) if available]

            if not available_indices or not env.edge_nodes[agent_id].new_task:
                optimal_actions.append(available_indices[-1] if available_indices else self.args.edge_node_num)
                continue

            task = env.edge_nodes[agent_id].new_task

            best_action = available_indices[-1]  # Default idle
            best_score = float('-inf')

            # Prioritize local execution
            if agent_id in available_indices:
                local_score = self._calculate_conservative_score(
                    env, agent_id, agent_id, task, conservativeness
                )
                if local_score > best_score:
                    best_score = local_score
                    best_action = agent_id

            # Check remote execution options
            for action_id in available_indices[:-1]:
                if action_id < len(env.edge_nodes) and action_id != agent_id:
                    score = self._calculate_conservative_score(
                        env, agent_id, action_id, task, conservativeness
                    )

                    # Remote execution needs higher score to be selected
                    if score > best_score + 0.1 * conservativeness:
                        best_score = score
                        best_action = action_id

            optimal_actions.append(best_action)

        return optimal_actions

    def _calculate_conservative_score(self, env, source_id, target_id, task, conservativeness):
        """
        Calculate conservative score - parameter optimization version
        """
        try:
            source_node = env.edge_nodes[source_id]
            target_node = env.edge_nodes[target_id]

            # 1. Reliability score (highest weight)
            reliability_score = self._get_optimized_reliability_score(
                source_node, target_node, task, source_id, target_id
            )

            # 2. Time margin score (second highest weight)
            time_score = self._get_optimized_time_score(
                source_node, target_node, task, source_id, target_id, conservativeness
            )

            # 3. Load score (lower weight)
            load_score = self._get_optimized_load_score(target_node)

            # 4. Efficiency score (lowest weight)
            efficiency_score = self._get_optimized_efficiency_score(target_node, task)

            # Strict feasibility check
            if reliability_score < self.threshold_params['reliability_threshold']:
                return float('-inf')
            if time_score == 0:
                return float('-inf')

            # Local execution bonus
            local_bonus = 0
            if source_id == target_id:
                local_bonus = self.strategy_params['local_preference_bonus'] * conservativeness

            # Weighted score
            total_score = (
                    self.optimization_weights['reliability_weight'] * reliability_score +
                    self.optimization_weights['deadline_weight'] * time_score +
                    self.optimization_weights['load_balance_weight'] * load_score +
                    self.optimization_weights['efficiency_weight'] * efficiency_score +
                    local_bonus
            )

            return total_score

        except Exception as e:
            return float('-inf')

    def _get_optimized_reliability_score(self, source_node, target_node, task, source_id, target_id):
        """
        Optimized reliability score
        """
        try:
            if source_id == target_id:
                # Local execution
                execution_time = task.task_cpu_cycle / target_node.cpu_capacity
                execution_reliability = math.exp(-target_node.execution_failure_rate * execution_time)
                system_reliability = execution_reliability
            else:
                # Remote execution
                transmission_time = task.task_size / source_node.transmission_rates[target_id]
                execution_time = task.task_cpu_cycle / target_node.cpu_capacity

                transmission_reliability = math.exp(
                    -source_node.transmission_failure_rates[target_id] * transmission_time)
                execution_reliability = math.exp(-target_node.execution_failure_rate * execution_time)
                system_reliability = transmission_reliability * execution_reliability

            # Apply failure penalty factor
            if system_reliability < self.threshold_params['reliability_threshold']:
                penalty = self.strategy_params['failure_penalty_factor']
                system_reliability = max(0.0, system_reliability - penalty * (
                            self.threshold_params['reliability_threshold'] - system_reliability))

            return system_reliability

        except Exception as e:
            return 0.0

    def _get_optimized_time_score(self, source_node, target_node, task, source_id, target_id, conservativeness):
        """
        Optimized time score
        """
        try:
            if source_id == target_id:
                # Local execution
                execution_time = task.task_cpu_cycle / target_node.cpu_capacity
                # Conservative estimate of waiting time
                queue_factor = self.strategy_params['conservative_factor'] * conservativeness
                waiting_time = len(target_node.execution_queue) * execution_time * queue_factor / max(target_node.k, 1)
                total_time = execution_time + waiting_time
            else:
                # Remote execution
                transmission_time = task.task_size / source_node.transmission_rates[target_id]
                execution_time = task.task_cpu_cycle / target_node.cpu_capacity
                queue_factor = self.strategy_params['conservative_factor'] * conservativeness
                waiting_time = len(target_node.execution_queue) * execution_time * queue_factor / max(target_node.k, 1)
                total_time = transmission_time + execution_time + waiting_time

            # Calculate time margin
            time_margin = task.task_deadline - total_time
            required_margin = task.task_deadline * self.threshold_params['min_time_margin'] * conservativeness

            if time_margin < required_margin:
                return 0.0  # Not enough time

            # Time score: the greater the margin, the higher the score
            time_score = min(1.0, time_margin / task.task_deadline)

            # Apply time pressure factor
            if time_score < 0.5:
                time_score *= self.strategy_params['time_pressure_factor']

            return time_score

        except Exception as e:
            return 0.0

    def _get_optimized_load_score(self, target_node):
        """
        Optimized load score
        """
        try:
            current_load = len(target_node.execution_queue) + len(target_node.executing_queue)
            max_load = (target_node.execution_queue_len + target_node.k) * self.threshold_params[
                'max_queue_utilization']

            if current_load >= max_load:
                return 0.0  # Load too high

            utilization = current_load / max_load
            load_score = 1.0 - utilization

            # Apply queue penalty
            if current_load > 0:
                queue_penalty = current_load * self.strategy_params['queue_penalty_factor']
                load_score = max(0.0, load_score - queue_penalty)

            return load_score

        except Exception as e:
            return 0.5

    def _get_optimized_efficiency_score(self, target_node, task):
        """
        Optimized efficiency score
        """
        try:
            execution_time = task.task_cpu_cycle / target_node.cpu_capacity
            optimal_time = self.args.mini_time_slot * 3  # Expected execution time

            if execution_time <= optimal_time:
                efficiency_score = 1.0
            else:
                efficiency_score = max(0.2, optimal_time / execution_time)

            # CPU core bonus
            cpu_bonus = min(0.2, target_node.cpu_core_num / 32.0)  # Maximum 32 cores
            efficiency_score += cpu_bonus

            return min(1.0, efficiency_score)

        except Exception as e:
            return 0.5

    def _fallback_search(self, env):
        """
        Fallback search strategy
        """
        env_copy = copy.deepcopy(env)
        env_copy.reset()

        actions_sequence = []
        total_reward = 0

        for step in range(self.max_search_depth):
            optimal_actions = self._simple_greedy_selection(env_copy)
            actions_sequence.append(optimal_actions)

            reward, terminated, info = env_copy.step(optimal_actions)
            total_reward += reward

            if terminated:
                break

        return actions_sequence, total_reward

    def _simple_greedy_selection(self, env):
        """
        Simple greedy selection
        """
        avail_actions = env.get_avail_actions()
        n_agents = len(avail_actions)
        optimal_actions = []

        for agent_id in range(n_agents):
            available_indices = [j for j, available in enumerate(avail_actions[agent_id]) if available]

            if not available_indices or not env.edge_nodes[agent_id].new_task:
                optimal_actions.append(available_indices[-1] if available_indices else self.args.edge_node_num)
                continue

            # Simple strategy: prioritize local execution, if local queue is full, select the idle node with the strongest CPU
            if agent_id in available_indices:
                local_load = len(env.edge_nodes[agent_id].execution_queue) + len(
                    env.edge_nodes[agent_id].executing_queue)
                if local_load < env.edge_nodes[agent_id].k:
                    optimal_actions.append(agent_id)
                    continue

            # Select the available node with the strongest CPU
            best_action = available_indices[-1]
            best_cpu = 0

            for action_id in available_indices[:-1]:
                if action_id < len(env.edge_nodes):
                    node = env.edge_nodes[action_id]
                    current_load = len(node.execution_queue) + len(node.executing_queue)
                    if current_load < node.k and node.cpu_core_num > best_cpu:
                        best_cpu = node.cpu_core_num
                        best_action = action_id

            optimal_actions.append(best_action)

        return optimal_actions

    def save_optimal_episode(self, env, actions_sequence):
        """Save optimal episode data"""
        try:
            env_copy = copy.deepcopy(env)
            env_copy.reset()

            obs_list = []
            actions_list = []
            rewards_list = []
            avail_actions_list = []
            next_obs_list = []
            masks_list = []

            total_reward = 0

            for step, actions in enumerate(actions_sequence):
                obs = env_copy.get_obs()
                avail_actions = env_copy.get_avail_actions()

                obs_list.append(obs.copy())
                actions_list.append(actions.copy())
                avail_actions_list.append(avail_actions.copy())

                reward, terminated, info = env_copy.step(actions)
                rewards_list.append(reward)
                total_reward += reward

                next_obs = env_copy.get_obs() if not terminated else obs.copy()
                next_obs_list.append(next_obs.copy())

                mask = 0.0 if terminated else 1.0
                masks_list.append(mask)

                if terminated:
                    break

            self.optimal_episode_data = {
                'obs': obs_list,
                'actions': actions_list,
                'rewards': rewards_list,
                'avail_actions': avail_actions_list,
                'next_obs': next_obs_list,
                'masks': masks_list,
                'total_reward': total_reward,
                'episode_info': info
            }

        except Exception as e:
            print(f"Error occurred while saving optimal episode data: {e}")

    def get_optimal_buffer_data(self):
        """Get optimal buffer data"""
        return self.optimal_episode_data

    def train(self):
        """Training method (not needed for optimal agent)"""
        pass