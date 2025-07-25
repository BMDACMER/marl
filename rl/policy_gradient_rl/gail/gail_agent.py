import torch as th
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
import os
import numpy as np

from buffer.ppo_expert_buffer import PPOExpertBuffer
from rl.policy_gradient_rl.gail.gail_network import Actor, Critic, Discriminator
from utils.action_selectors import soft_selector, greedy_selector
from utils.advantage_utils import get_gae
from optimal.optimal_agent import OptimalAgent


class GAILAgent:

    def __init__(self, args):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.args = args

        self.actor = Actor(args).to(self.device)
        self.critic = Critic(args).to(self.device)
        self.discriminator = Discriminator(args).to(self.device)
        
        self.actor_optimizer = Adam(params=self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = Adam(params=self.critic.parameters(), lr=args.lr)
        self.discriminator_optimizer = Adam(params=self.discriminator.parameters(), lr=args.lr)

        self.buffer = PPOExpertBuffer(args)
        self.expert_agent = OptimalAgent(args)

        self.bc_loss_weight = getattr(args, 'bc_loss_weight', 1.0)
        self.bc_decay_rate = getattr(args, 'bc_decay_rate', 0.99)
        self.min_bc_weight = getattr(args, 'min_bc_weight', 0.1)

        self.expert_obs_buffer = []
        self.expert_actions_buffer = []
        self.max_expert_buffer_size = args.max_expert_buffer_size

        self.update_count = 0
        self.performance_history = []

        print(f"Initialize the GAIL agent, BC weights: {self.bc_loss_weight}")

    @th.no_grad()
    def select_actions(self, obs, avail_actions, test_mode=False):
        obs = obs.to(self.device)
        agent_ids = th.eye(self.args.n_agents).unsqueeze(0).expand(self.args.n_threads, -1, -1).to(self.device)
        obs = th.cat([obs, agent_ids], dim=-1)
        avail_actions = avail_actions.to(self.device)
        actor_outs = self.actor(obs)
        actor_outs[avail_actions == 0] = -1e10

        if test_mode:
            picked_actions = greedy_selector(actor_outs)
        else:
            picked_actions = soft_selector(actor_outs)
        return picked_actions

    def get_expert_actions(self, env):
        try:
            return self.expert_agent.get_current_optimal_action(env)
        except Exception as e:
            print(f"Failed to acquire the expert action: {e}")
            return None

    def get_expert_ratio(self, t_env):
        if t_env < self.args.bc_guidance_steps:
            return 1.0
        elif t_env < self.args.expert_guidance_steps:
            progress = (t_env - self.args.bc_guidance_steps) / (self.args.expert_guidance_steps - self.args.bc_guidance_steps)
            return max(0.2, 0.9 * (1 - progress))
        else:
            return 0.2

    def store_expert_data(self, obs, actions):
        self.expert_obs_buffer.append(obs.clone().cpu())
        self.expert_actions_buffer.append(actions.clone().cpu())

        if len(self.expert_obs_buffer) > self.max_expert_buffer_size:
            self.expert_obs_buffer.pop(0)
            self.expert_actions_buffer.pop(0)

    def compute_bc_loss(self, obs, avail_actions, actions):
        if len(self.expert_obs_buffer) == 0:
            return th.tensor(0.0, device=self.device)

        sample_size = min(64, len(self.expert_obs_buffer))
        indices = np.random.choice(len(self.expert_obs_buffer), sample_size, replace=False)

        expert_obs_batch = th.stack([self.expert_obs_buffer[i] for i in indices]).to(self.device)
        expert_actions_batch = th.stack([self.expert_actions_buffer[i] for i in indices]).to(self.device)

        agent_ids = th.eye(self.args.n_agents).unsqueeze(0).expand(sample_size, -1, -1).to(self.device)
        expert_obs_batch = th.cat([expert_obs_batch, agent_ids], dim=-1)

        actor_outs = self.actor(expert_obs_batch)
        log_probs = F.log_softmax(actor_outs, dim=-1)

        bc_loss = F.nll_loss(
            log_probs.view(-1, self.args.n_actions),
            expert_actions_batch.view(-1),
            reduction='mean'
        )

        return bc_loss

    def evaluate_actions(self, inputs, avail_actions, actions):
        actor_outs = self.actor(inputs)
        actor_outs[avail_actions == 0] = -1e10
        probs = F.softmax(actor_outs, dim=-1)
        dist = Categorical(probs)
        log_probs_taken = dist.log_prob(actions.squeeze(-1))
        entropy = dist.entropy()
        return log_probs_taken.unsqueeze(-1), entropy.unsqueeze(-1)

    def update_bc_weight(self):
        if len(self.performance_history) >= 5:
            recent_performance = np.mean(self.performance_history[-5:])
            if recent_performance < 0.7:
                self.bc_loss_weight = min(2.0, self.bc_loss_weight * 1.1)
            else:
                self.bc_loss_weight = max(self.min_bc_weight, self.bc_loss_weight * self.bc_decay_rate)
        else:
            self.bc_loss_weight = max(self.min_bc_weight, self.bc_loss_weight * self.bc_decay_rate)

    def train(self):
        obs, avail_actions, actions, rewards, masks, next_obs = self.buffer.sample()
        
        if len(self.expert_obs_buffer) >= 10:
            try:
                agent_ids = th.eye(self.args.n_agents).unsqueeze(0).unsqueeze(0).expand(
                    obs.size(0), obs.size(1), -1, -1).to(self.device)
                obs_with_id = th.cat([obs, agent_ids], dim=-1)
                actions_one_hot = F.one_hot(actions.squeeze(-1), num_classes=self.args.n_actions).float()
                
                sample_size = min(64, len(self.expert_obs_buffer))
                indices = np.random.choice(len(self.expert_obs_buffer), sample_size, replace=False)
                
                expert_obs = th.stack([self.expert_obs_buffer[i] for i in indices]).to(self.device)
                expert_actions = th.stack([self.expert_actions_buffer[i] for i in indices]).to(self.device)
                
                expert_agent_ids = th.eye(self.args.n_agents).unsqueeze(0).expand(sample_size, -1, -1).to(self.device)
                expert_obs_with_id = th.cat([expert_obs, expert_agent_ids], dim=-1)
                expert_actions_one_hot = F.one_hot(expert_actions, num_classes=self.args.n_actions).float()
                
                expert_logits = self.discriminator(expert_obs_with_id, expert_actions_one_hot)
                agent_logits = self.discriminator(obs_with_id.reshape(-1, obs_with_id.size(-1)), 
                                               actions_one_hot.reshape(-1, actions_one_hot.size(-1)))
                agent_logits = agent_logits.reshape(obs.size(0), obs.size(1), obs.size(2), 1)
                
                bce_loss = nn.BCEWithLogitsLoss()
                discriminator_loss = bce_loss(agent_logits.reshape(-1, 1), 
                                             th.ones_like(agent_logits.reshape(-1, 1))) + \
                                    bce_loss(expert_logits, 
                                            th.zeros_like(expert_logits))
                
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                if self.args.use_grad_clip:
                    th.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.args.grad_norm_clip)
                self.discriminator_optimizer.step()
                
                with th.no_grad():
                    agent_logits = self.discriminator(obs_with_id.reshape(-1, obs_with_id.size(-1)), 
                                                   actions_one_hot.reshape(-1, actions_one_hot.size(-1)))
                    agent_logits = agent_logits.reshape(obs.size(0), obs.size(1), obs.size(2), 1)
                    
                    gail_rewards = -F.logsigmoid(agent_logits) + F.logsigmoid(-agent_logits)
                    
                    rewards = gail_rewards
            except Exception as e:
                print(f"GAIL trains incorrectly, using raw rewards: {e}")
        
        # 添加agent ID
        agent_ids = th.eye(self.args.n_agents).unsqueeze(0).unsqueeze(0).expand(
            obs.size(0), obs.size(1), -1, -1).to(self.device)
        obs = th.cat([obs, agent_ids], dim=-1)
        next_obs = th.cat([next_obs, agent_ids], dim=-1)

        if self.args.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        old_log_pi_taken, old_entropy = self.evaluate_actions(obs, avail_actions, actions)
        with th.no_grad():
            old_v = self.critic(obs)
            old_next_v = self.critic(next_obs)
            advantages = get_gae(self.args, rewards, old_v, old_next_v, masks)

        if self.args.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        targets = advantages + old_v

        self.update_bc_weight()

        for k in range(self.args.epochs):
            log_pi_taken, entropy = self.evaluate_actions(obs, avail_actions, actions)
            v = self.critic(obs)
            critic_loss = (((v - targets) * masks) ** 2).sum() / masks.sum()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.args.use_grad_clip:
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_norm_clip)
            self.critic_optimizer.step()

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

            if self.args.use_entropy:
                ppo_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * masks).sum() / masks.sum()
            else:
                ppo_loss = -((th.min(surr1, surr2)) * masks).sum() / masks.sum()

            bc_loss = self.compute_bc_loss(obs, avail_actions, actions)

            total_actor_loss = ppo_loss + self.bc_loss_weight * bc_loss

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            if self.args.use_grad_clip:
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
            self.actor_optimizer.step()

        self.update_count += 1

        if self.update_count % 100 == 0:
            print(f"Update {self.update_count}: BC weight = {self.bc_loss_weight:.4f}, expert data size = {len(self.expert_obs_buffer)}")

    def update_performance_monitoring(self, success_rate):
        self.performance_history.append(success_rate)
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)

    def save_models(self, path):
        os.makedirs(path, exist_ok=True)
        th.save(self.actor.state_dict(), "{}/actor.th".format(path))
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.discriminator.state_dict(), "{}/discriminator.th".format(path))

        if self.expert_obs_buffer:
            th.save({
                'expert_obs': self.expert_obs_buffer,
                'expert_actions': self.expert_actions_buffer
            }, "{}/expert_data.th".format(path))

    def load_models(self, path):
        self.actor.load_state_dict(th.load("{}/actor.th".format(path), map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        
        disc_path = "{}/discriminator.th".format(path)
        if os.path.exists(disc_path):
            self.discriminator.load_state_dict(th.load(disc_path, map_location=lambda storage, loc: storage))

        expert_data_path = "{}/expert_data.th".format(path)
        if os.path.exists(expert_data_path):
            expert_data = th.load(expert_data_path, map_location=lambda storage, loc: storage)
            self.expert_obs_buffer = expert_data['expert_obs']
            self.expert_actions_buffer = expert_data['expert_actions']

    def load_offline_models(self, path):
        self.load_models(path)