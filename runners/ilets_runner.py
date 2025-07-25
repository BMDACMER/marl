import torch as th
import numpy as np
import os
import time
from collections import defaultdict

from runners.episode_runner import EpisodeRunner


class ILETSRunner(EpisodeRunner):

    def __init__(self, args, agent):
        super().__init__(args, agent)
        
        self.expert_collection_frequency = getattr(args, 'expert_collection_frequency', 10)
        self.expert_guidance_steps = getattr(args, 'expert_guidance_steps', 30000)
        self.bc_guidance_steps = getattr(args, 'bc_guidance_steps', 8000)
        
        self.recent_success_rates = []
        self.expert_data_collected = 0
        self.last_expert_collection = 0
        
        self.training_frequency = getattr(args, 'training_frequency', 5)
        self.episode_count = 0
        
        self.best_success_rate = 0.0
        self.best_model_path = os.path.join(getattr(args, 'results_dir', './results'), 'best_model')

    def run(self, test_mode=False):
        from loguru import logger

        # -------- test mode --------
        if test_mode:
            return super().run(test_mode=True)

        # -------- train mode（single episode） --------
        episode_info = self.run_single_episode()

        self.episode_count += 1 if hasattr(self, 'episode_count') else 1

        if self.should_collect_expert_data(self.episode_count):
            self.collect_expert_data()

        if self.episode_count % 1 == 0:
            if self.agent.buffer.can_sample():
                self.agent.train()

        return episode_info

    def run_single_episode(self):
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
        step_count = 0
        
        while not terminated and step_count < self.args.episode_limit:
            obs = th.as_tensor(self.env.get_obs(), dtype=th.float)
            avail_actions = th.as_tensor(self.env.get_avail_actions(), dtype=th.int)
            
            actions = self.agent.select_actions(obs.unsqueeze(0), avail_actions.unsqueeze(0), test_mode=False)
            cpu_actions = actions.to("cpu").numpy()
            
            reward, terminated, info = self.env.step(cpu_actions[0])
            
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
            
            rewards = th.as_tensor(reward).unsqueeze(0).unsqueeze(0).repeat(self.args.n_agents, 1)
            masks = th.as_tensor(1 - int(terminated)).unsqueeze(0).repeat(self.args.n_agents, 1)
            next_obs = th.as_tensor(self.env.get_obs(), dtype=th.float)
            
            self.agent.buffer.insert(
                obs.unsqueeze(0), 
                avail_actions.unsqueeze(0), 
                actions.unsqueeze(0).unsqueeze(-1), 
                rewards.unsqueeze(0), 
                masks.unsqueeze(0), 
                next_obs.unsqueeze(0)
            )
            
            self.t_env += 1
            step_count += 1
        
        episode_info = {
            "episode_return": episode_return,
            "success_rate": success_finish_task_number / max(finish_task_number, 1),
            "drop_rate": drop_task_number / max(finish_task_number, 1),
            "failure_rate": failure_task_number / max(finish_task_number, 1),
            "task_completion_time": task_completion_time / max(success_finish_task_number + drop_task_number, 1),
            "max_hop_dict": max_hop_dict
        }
        
        return episode_info

    def should_collect_expert_data(self, episode_count):
        if self.t_env - self.last_expert_collection < 50:
            return False
            
        if self.t_env < self.bc_guidance_steps:
            return episode_count % max(1, self.expert_collection_frequency // 3) == 0
        elif self.t_env < self.expert_guidance_steps:
            if len(self.recent_success_rates) >= 3:
                recent_avg = np.mean(self.recent_success_rates[-3:])
                if recent_avg > 0.85:
                    return episode_count % (self.expert_collection_frequency * 2) == 0
                else:
                    return episode_count % self.expert_collection_frequency == 0
            return episode_count % self.expert_collection_frequency == 0
        else:
            if len(self.recent_success_rates) >= 5:
                recent_avg = np.mean(self.recent_success_rates[-3:])
                recent_5 = self.recent_success_rates[-5:]
                trend = np.polyfit(range(5), recent_5, 1)[0] if len(recent_5) == 5 else 0
                
                if trend < -0.015 and recent_avg < 0.82:
                    return episode_count % max(5, self.expert_collection_frequency // 3) == 0
                elif recent_avg < 0.8:
                    return episode_count % (self.expert_collection_frequency // 2) == 0
                else:
                    return episode_count % (self.expert_collection_frequency * 3) == 0
            return episode_count % (self.expert_collection_frequency * 2) == 0

    def collect_expert_data(self):
        try:
            # print(f"collect expert data (step: {self.t_env})")
            
            current_env_state = self.env
            
            from envs.env_register import env_register
            expert_env = env_register[self.args.env_name](self.args)
            expert_env.reset()
            
            terminated = False
            episode_reward = 0
            step_count = 0
            collected_steps = 0
            
            while not terminated and step_count < self.args.episode_limit:
                obs = expert_env.get_obs()
                avail_actions = expert_env.get_avail_actions()
                
                expert_actions = self.agent.get_expert_actions(expert_env)
                
                valid_actions = True
                for i, action in enumerate(expert_actions):
                    if action >= len(avail_actions[i]) or not avail_actions[i][action]:
                        valid_actions = False
                        break
                
                if not valid_actions:
                    print("invalid expert actions, skip this step")
                    step_count += 1
                    continue
                
                reward, terminated, info = expert_env.step(expert_actions)
                episode_reward += reward
                
                obs_tensor = th.tensor(obs, dtype=th.float32)
                expert_actions_tensor = th.tensor(expert_actions, dtype=th.long)
                
                self.agent.store_expert_data(obs_tensor, expert_actions_tensor, reward)
                collected_steps += 1
                step_count += 1
            
            self.expert_data_collected += 1
            self.last_expert_collection = self.t_env
            
            expert_env.close()
            
        except Exception as e:
            print(f"Failed expert data collection: {e}")

    def perform_test_and_log(self):
        from loguru import logger

        test_results, success_rates = [], []
        for _ in range(self.args.test_nepisode):
            epi_info = super().run(test_mode=True)
            test_results.append(epi_info["episode_return"])
            success_rates.append(epi_info["success_rate"])

        mean_reward = float(np.mean(test_results))
        mean_success_rate = float(np.mean(success_rates))

        self.recent_success_rates.append(mean_success_rate)
        if len(self.recent_success_rates) > 10:
            self.recent_success_rates.pop(0)
        self.agent.update_performance_monitoring(mean_success_rate)

        logger.info(f"[Eval] step={self.t_env} reward={mean_reward:.4f} succ={mean_success_rate:.4f} bc={self.agent.bc_loss_weight:.4f} exp={len(self.agent.expert_obs_buffer)}")

        self._update_best_model(mean_success_rate)

    def save_models(self):
        """Save the model and training state"""
        save_path = os.path.join(getattr(self.args, 'results_dir', './results'), "models")
        os.makedirs(save_path, exist_ok=True)
        
        self.agent.save_models(save_path)
        
        print(f"The model has been saved to : {save_path}")

    def cleanup_training(self):

        if self.recent_success_rates:
            final_success_rate = np.mean(self.recent_success_rates[-3:])
            print(f"  Final success rate: {final_success_rate:.4f}")
            
            if len(self.recent_success_rates) >= 5:
                initial_success_rate = np.mean(self.recent_success_rates[:3])
                improvement = final_success_rate - initial_success_rate

    def _update_best_model(self, current_success_rate):

        if current_success_rate > self.best_success_rate:
            self.best_success_rate = float(current_success_rate)
            os.makedirs(self.best_model_path, exist_ok=True)
            self.agent.save_models(self.best_model_path)
            print(f"  ↑  Success rate increased to {self.best_success_rate:.4f}，the best model has been saved")

        # Prevent later performance collapse
        performance_drop = self.best_success_rate - current_success_rate
        late_stage = self.t_env > self.expert_guidance_steps
        if late_stage and performance_drop > 0.05 and os.path.isdir(self.best_model_path):
            print("  ↓ Significant performance degradation was monitored, rollback to the best model and enforcing BC constraints")
            self.agent.load_models(self.best_model_path)
            self.agent.bc_loss_weight = max(self.agent.bc_loss_weight, 1.0)
