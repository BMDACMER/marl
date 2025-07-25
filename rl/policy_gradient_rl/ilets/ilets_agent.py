import torch as th
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

from buffer.episode_buffer import EpisodeBuffer
from rl.policy_gradient_rl.ilets.ilets_network import Actor, Critic
from utils.action_selectors import soft_selector, greedy_selector
from utils.advantage_utils import get_returns
from optimal.optimal_agent import OptimalAgent


class ILETSAgent:
    def __init__(self, args):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.args = args

        self.actor = Actor(args).to(self.device)
        self.critic = Critic(args).to(self.device)
        
        self.actor_optimizer = Adam(params=self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = Adam(params=self.critic.parameters(), lr=args.lr * 2)
        
        self.buffer = EpisodeBuffer(args)
        self.expert_agent = OptimalAgent(args)

        self.bc_loss_weight = getattr(args, 'bc_loss_weight', 2.0)
        self.bc_decay_rate = getattr(args, 'bc_decay_rate', 0.9995)
        self.min_bc_weight = getattr(args, 'min_bc_weight', 0.8)
        self.imitation_threshold = getattr(args, 'imitation_threshold', 0.75)
        
        self.expert_obs_buffer = []
        self.expert_actions_buffer = []
        self.expert_rewards_buffer = []
        self.max_expert_buffer_size = getattr(args, 'max_expert_buffer_size', 1000)
        
        self.training_step = 0
        self.success_rate_history = []
        self.last_bc_loss = 0.0
        
        self.value_loss_coeff = getattr(args, 'value_loss_coeff', 0.5)
        self.entropy_coeff = getattr(args, 'entropy_coeff', 0.01)
        
        self.performance_window = []
        self.stability_threshold = 0.03
        self.consecutive_drops = 0

    @th.no_grad()
    def select_actions(self, obs, avail_actions, test_mode=False):
        obs = obs.to(self.device)
        avail_actions = avail_actions.to(self.device)
        batch_size = obs.size(0)
        agent_ids = th.eye(self.args.n_agents).unsqueeze(0).expand(
            batch_size, -1, -1
        ).to(self.device)
        obs_with_id = th.cat([obs, agent_ids], dim=-1)
        actor_outs = self.actor(obs_with_id)
        actor_outs[avail_actions == 0] = -1e10
        if test_mode:
            picked_actions = greedy_selector(actor_outs)
        else:
            picked_actions = soft_selector(actor_outs)
            
        return picked_actions

    def get_expert_actions(self, env):
        try:
            expert_actions = self.expert_agent.get_current_optimal_action(env)
            if expert_actions is None or len(expert_actions) != self.args.n_agents:
                # 兜底策略
                avail_actions = env.get_avail_actions()
                expert_actions = []
                for i in range(self.args.n_agents):
                    available_indices = [j for j, available in enumerate(avail_actions[i]) if available]
                    if available_indices:
                        expert_actions.append(np.random.choice(available_indices))
                    else:
                        expert_actions.append(self.args.n_agents)
                expert_actions = np.array(expert_actions)
            return expert_actions
        except Exception as e:
            print(f"Failed to acquire the expert action: {e}")
            return np.arange(self.args.n_agents)

    def store_expert_data(self, obs, actions, rewards):
        if obs.dim() == 3:
            obs = obs[0]
        if actions.dim() > 1:
            actions = actions.squeeze()
            
        # Data validation for debug
        if obs.shape[0] != self.args.n_agents or actions.shape[0] != self.args.n_agents:
            print(f"The expert data dimension does not match: obs={obs.shape}, actions={actions.shape}")
            return
            
        self.expert_obs_buffer.append(obs.clone().cpu())
        self.expert_actions_buffer.append(actions.clone().cpu())
        self.expert_rewards_buffer.append(float(rewards))
        
        if len(self.expert_obs_buffer) > self.max_expert_buffer_size:
            self.expert_obs_buffer.pop(0)
            self.expert_actions_buffer.pop(0)
            self.expert_rewards_buffer.pop(0)

    def compute_bc_loss(self):
        if len(self.expert_obs_buffer) < 5:
            return th.tensor(0.0, device=self.device)
        
        try:
            sample_size = min(32, len(self.expert_obs_buffer))
            indices = np.random.choice(len(self.expert_obs_buffer), sample_size, replace=False)
            
            expert_obs_list = [self.expert_obs_buffer[i] for i in indices]
            expert_actions_list = [self.expert_actions_buffer[i] for i in indices]
            
            valid_data = []
            valid_actions = []
            for obs, actions in zip(expert_obs_list, expert_actions_list):
                if obs.shape[0] == self.args.n_agents and actions.shape[0] == self.args.n_agents:
                    valid_data.append(obs)
                    valid_actions.append(actions)
            
            if len(valid_data) < 3:
                return th.tensor(0.0, device=self.device)
                    
            expert_obs_batch = th.stack(valid_data).to(self.device)
            expert_actions_batch = th.stack(valid_actions).to(self.device)
            
            batch_size = expert_obs_batch.size(0)
            agent_ids = th.eye(self.args.n_agents).unsqueeze(0).expand(
                batch_size, -1, -1
            ).to(self.device)
            expert_obs_with_id = th.cat([expert_obs_batch, agent_ids], dim=-1)
            
            actor_outs = self.actor(expert_obs_with_id)
            log_probs = F.log_softmax(actor_outs, dim=-1)
            log_probs_flat = log_probs.view(-1, self.args.n_actions)
            expert_actions_flat = expert_actions_batch.view(-1)
            bc_loss = F.nll_loss(log_probs_flat, expert_actions_flat, reduction='mean')
            self.last_bc_loss = bc_loss.item()

            return bc_loss
            
        except Exception as e:
            print(f"The BC loss calculation failed: {e}")
            return th.tensor(0.0, device=self.device)

    def evaluate_actions(self, inputs, avail_actions, actions):
        actor_outs = self.actor(inputs)
        actor_outs[avail_actions == 0] = -1e10
        
        probs = F.softmax(actor_outs, dim=-1)
        log_probs = F.log_softmax(actor_outs, dim=-1)
        log_probs_taken = log_probs.gather(-1, actions)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        
        return log_probs_taken, entropy

    def update_bc_weight(self):
        old_weight = self.bc_loss_weight
        self.bc_loss_weight = max(self.min_bc_weight, 
                                  self.bc_loss_weight * self.bc_decay_rate)
        
        if len(self.success_rate_history) >= 5:
            recent_5 = self.success_rate_history[-5:]
            recent_3 = self.success_rate_history[-3:]
            current_avg = np.mean(recent_3)
            
            if len(recent_5) >= 5:
                trend = np.polyfit(range(5), recent_5, 1)[0]  # 线性拟合斜率
                if trend < -0.01 and current_avg < 0.82:
                    self.consecutive_drops += 1
                    boost_factor = 1.2 + 0.1 * min(self.consecutive_drops, 3)
                    self.bc_loss_weight = min(4.0, self.bc_loss_weight * boost_factor)
                else:
                    self.consecutive_drops = max(0, self.consecutive_drops - 1)
            
            if current_avg > 0.87 and self.consecutive_drops == 0:
                self.bc_loss_weight = max(self.min_bc_weight, 
                                          self.bc_loss_weight * 0.995)
            elif current_avg < self.imitation_threshold:
                self.bc_loss_weight = min(3.5, self.bc_loss_weight * 1.15)

        if self.last_bc_loss < 0.005 and len(self.expert_obs_buffer) > 10:
            self.bc_loss_weight = min(3.0, self.bc_loss_weight * 1.2)

        if self.training_step > 30000:
            self.bc_loss_weight = max(1.0, self.bc_loss_weight)

    def train(self):
        obs, avail_actions, actions, rewards, masks, next_obs = self.buffer.sample()
        batch_size, episode_len = obs.shape[0], obs.shape[1]
        agent_ids = th.eye(self.args.n_agents).unsqueeze(0).unsqueeze(0).expand(
            batch_size, episode_len, -1, -1
        ).to(self.device)
        obs_with_id = th.cat([obs, agent_ids], dim=-1)
        
        if self.args.normalize_rewards and rewards.std() > 1e-6:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        returns = get_returns(self.args, rewards, masks)
        values = self.critic(obs_with_id)  # [batch, episode_len, n_agents, 1]
        
        if returns.shape != values.shape:
            if len(returns.shape) == 3:  # [batch, episode_len, n_agents]
                returns = returns.unsqueeze(-1)  # [batch, episode_len, n_agents, 1]
        advantages = (returns - values).detach()
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        log_probs_taken, entropy = self.evaluate_actions(obs_with_id, avail_actions, actions)
        actor_loss = -(log_probs_taken * advantages * masks).sum() / masks.sum()
        entropy_loss = -self.entropy_coeff * (entropy * masks).sum() / masks.sum()
        
        bc_loss = self.compute_bc_loss()
        
        total_actor_loss = actor_loss + entropy_loss + self.bc_loss_weight * bc_loss
        
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        
        if hasattr(self.args, 'use_grad_clip') and self.args.use_grad_clip:
            grad_norm = th.nn.utils.clip_grad_norm_(self.actor.parameters(), 
                                                    getattr(self.args, 'grad_norm_clip', 5.0))
            if grad_norm > 10.0:
                print(f"  [Warning] Gradient explosion detected: {grad_norm:.2f}, skip this update.")
                return
        else:
            total_norm = 0
            for p in self.actor.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            if total_norm > 15.0:
                print(f"  [Warning] gradient explosion detected: {total_norm:.2f}, scaling gradients")
                for p in self.actor.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(5.0 / total_norm)
        
        self.actor_optimizer.step()
        
        values_flat = values.view(-1)
        returns_flat = returns.view(-1)
        masks_flat = masks.view(-1)
        
        valid_indices = masks_flat > 0
        if valid_indices.sum() > 0:
            value_loss = F.smooth_l1_loss(
                values_flat[valid_indices], 
                returns_flat[valid_indices], 
                reduction='mean'
            )
        else:
            value_loss = th.tensor(0.0, device=self.device)
        
        critic_loss = value_loss * self.value_loss_coeff
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        if hasattr(self.args, 'use_grad_clip') and self.args.use_grad_clip:
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), 
                                        getattr(self.args, 'grad_norm_clip', 5.0))
        self.critic_optimizer.step()
        self.training_step += 1
        if self.training_step % 15 == 0:
            self.update_bc_weight()

    def update_performance_monitoring(self, success_rate):
        self.success_rate_history.append(success_rate)
        if len(self.success_rate_history) > 30:
            self.success_rate_history.pop(0)

        self.performance_window.append(success_rate)
        if len(self.performance_window) > 10:
            self.performance_window.pop(0)

    def save_models(self, path):
        th.save(self.actor.state_dict(), f"{path}/actor.th")
        th.save(self.critic.state_dict(), f"{path}/critic.th")
        
        training_state = {
            'bc_loss_weight': self.bc_loss_weight,
            'training_step': self.training_step,
            'success_rate_history': self.success_rate_history
        }
        th.save(training_state, f"{path}/training_state.th")

    def load_models(self, path):
        self.actor.load_state_dict(
            th.load(f"{path}/actor.th", map_location=lambda storage, loc: storage)
        )
        self.critic.load_state_dict(
            th.load(f"{path}/critic.th", map_location=lambda storage, loc: storage)
        )
        
        try:
            training_state = th.load(f"{path}/training_state.th", 
                                     map_location=lambda storage, loc: storage)
            self.bc_loss_weight = training_state['bc_loss_weight']
            self.training_step = training_state['training_step']
            self.success_rate_history = training_state['success_rate_history']
            print(f"The training state is loaded, current BC weights: {self.bc_loss_weight}")
        except:
            print("Training status file not found, default configuration used.")
            
    def should_collect_expert_data(self, episode_count):
        if self.t_env < 5000:
            return episode_count % 3 == 0
        elif self.t_env < 20000:
            return episode_count % 5 == 0
        else:
            if len(self.recent_success_rates) >= 3:
                recent_avg = np.mean(self.recent_success_rates[-3:])
                if recent_avg < 0.8:
                    return episode_count % 8 == 0
            return episode_count % 15 == 0
            