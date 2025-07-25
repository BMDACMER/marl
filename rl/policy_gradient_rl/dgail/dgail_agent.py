import torch as th
import torch.nn.functional as F
from torch.optim import Adam
import os
import numpy as np
import math

from buffer.ppo_expert_buffer import PPOExpertBuffer
from rl.policy_gradient_rl.dgail.dgail_network import PolicyNet, ValueNet, DiffusionDiscriminator, GAILDiscriminator
from utils.action_selectors import soft_selector, greedy_selector
from utils.advantage_utils import get_gae
from optimal.optimal_agent import OptimalAgent


class RewardScaler:
    def __init__(self, shape, gamma=0.99):
        self.shape = shape
        self.gamma = gamma
        self.running_mean_std = RunningMeanStd(shape=self.shape)

    def __call__(self, x):
        self.running_mean_std.update(x.cpu().numpy())
        std_tensor = th.tensor(self.running_mean_std.std, dtype=x.dtype, device=x.device)
        return x / (std_tensor + 1e-8)


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)


class DGAILAgent:
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.device = args.device

        self.state_action_dim = self.obs_shape + self.n_actions
        self.policy_input_dim = self.obs_shape + self.n_agents
        
        self.policy = PolicyNet(self.policy_input_dim, self.n_actions, args.hidden_dim).to(self.device)
        self.value = ValueNet(self.obs_shape, args.hidden_dim).to(self.device)
        self.discriminator = DiffusionDiscriminator(
            input_dim=self.state_action_dim,
            hidden_dim=args.hidden_dim
        ).to(self.device)
        self.gail_discriminator = GAILDiscriminator(self.state_action_dim, args.hidden_dim).to(self.device)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=args.policy_lr)
        self.value_optimizer = Adam(self.value.parameters(), lr=args.value_lr)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=args.discriminator_lr)
        self.gail_discriminator_optimizer = Adam(self.gail_discriminator.parameters(), lr=args.gail_discriminator_lr)

        self.buffer = PPOExpertBuffer(args)
        self.expert_agent = OptimalAgent(args)

        self.expert_transitions = []
        self.max_expert_buffer_size = args.max_expert_buffer_size
        
        self.update_count = 0
        self.training_stats = {}
        
        self._bc_pretrain_steps = 700
        self._bc_optimizer = Adam(self.policy.parameters(), lr=1e-3)

        self._disc_update_count = 0

        self._r_lp = None

        self._success_ema = 0.0
        self._w_factor = 1.0
        self._target_success = 0.95

        self._target_success = 0.95
        self._simplified_reward = getattr(args, 'simplified_reward', True)

        self._use_simplified_reward = getattr(args, 'use_simplified_reward', True)

        self.bc_loss_weight = getattr(args, 'bc_loss_weight', 0.3)
        self.bc_decay_rate = getattr(args, 'bc_decay_rate', 0.995)
        self.min_bc_weight = getattr(args, 'min_bc_weight', 0.05)

        self.performance_history = []

    @th.no_grad()
    def select_actions(self, obs, avail_actions, test_mode=False):
        obs = obs.to(self.device)
        agent_ids = th.eye(self.args.n_agents).unsqueeze(0).expand(self.args.n_threads, -1, -1).to(self.device)
        obs_with_id = th.cat([obs, agent_ids], dim=-1)
        avail_actions = avail_actions.to(self.device)

        policy_logits = self.policy(obs_with_id)
        policy_logits[avail_actions == 0] = -1e10

        if test_mode:
            return greedy_selector(policy_logits)
        return soft_selector(policy_logits)

    def get_expert_actions(self, env):
        try:
            return self.expert_agent.get_current_optimal_action(env)
        except Exception as e:
            print(f"Failed to acquire the expert action: {e}")
            return None

    def store_expert_data(self, obs, actions):
        """
        Store expert (observation, action) pairs
        - obs: Expert observations (Tensor)
        - actions: Expert actions (Tensor)
        """
        # Ensure data is on CPU to avoid GPU memory usage
        obs_cpu = obs.clone().detach().cpu()
        actions_cpu = actions.clone().detach().cpu()
        
        self.expert_transitions.append((obs_cpu, actions_cpu))

        # Maintain buffer size
        if len(self.expert_transitions) > self.max_expert_buffer_size:
            self.expert_transitions.pop(0)

    def _compute_bc_loss(self):
        """Randomly sample from expert buffer and compute cross-entropy BC loss"""
        # if len(self.expert_transitions) < self.args.batch_expert_transitions:
        #     return th.tensor(0.0, device=self.device)

        sample_size = min(self.args.batch_expert_transitions, len(self.expert_transitions))
        idx = np.random.choice(len(self.expert_transitions), sample_size, replace=False)

        obs_batch = th.stack([self.expert_transitions[i][0] for i in idx]).to(self.device)
        act_batch = th.stack([self.expert_transitions[i][1] for i in idx]).to(self.device)

        agent_ids = th.eye(self.n_agents, device=self.device).unsqueeze(0).expand(sample_size, -1, -1)
        obs_batch = th.cat([obs_batch, agent_ids], dim=-1).reshape(-1, self.policy_input_dim)

        logits = self.policy(obs_batch)
        log_p = F.log_softmax(logits, dim=-1)
        bc_loss = F.nll_loss(log_p, act_batch.view(-1), reduction='mean')
        return bc_loss

    def _update_bc_weight(self):
        if len(self.performance_history) < 5:
            self.bc_loss_weight = max(self.min_bc_weight, self.bc_loss_weight * self.bc_decay_rate)
            return
        recent = np.mean(self.performance_history[-5:])
        if recent < 0.75:
            self.bc_loss_weight = min(2.0, self.bc_loss_weight * 1.1)
        else:
            self.bc_loss_weight = max(self.min_bc_weight, self.bc_loss_weight * self.bc_decay_rate)

    def _prepare_batch(self, obs, actions):
        """
        Prepare observation and action data into (state-action) pairs.
        [FIXED] Use more robust reshape(-1, ...) to handle inputs of different dimensions,
        ensuring all leading dimensions (threads, steps, agents) are properly flattened.
        """
        obs_tensor = th.as_tensor(obs, dtype=th.float32, device=self.device)
        actions_tensor = th.as_tensor(actions, dtype=th.long, device=self.device)

        # 展平obs，只保留最后一维（特征维度）
        obs_flat = obs_tensor.reshape(-1, self.args.obs_shape)
        
        # 展平actions为一维向量
        actions_flat = actions_tensor.reshape(-1)
        
        # 确保展平后的数量一致
        if obs_flat.shape[0] != actions_flat.shape[0]:
            raise ValueError(f"Mismatched number of transitions in obs ({obs_flat.shape[0]}) and actions ({actions_flat.shape[0]})")

        actions_one_hot = th.nn.functional.one_hot(actions_flat, num_classes=self.args.n_actions).float()
        
        return th.cat([obs_flat, actions_one_hot], dim=-1)

    def _update_discriminator(self, agent_sa):
        if len(self.expert_transitions) < self.args.batch_size_run:
            return 0.0, 0.0, 0.0 # Return default values if not training

        if not self.expert_transitions:
            return 0.0, 0.0, 0.0
        all_expert_obs = th.cat([t[0] for t in self.expert_transitions], dim=0)
        all_expert_actions = th.cat([t[1] for t in self.expert_transitions], dim=0)
        
        num_transitions = agent_sa.size(0)
        expert_indices = np.random.choice(all_expert_obs.shape[0], num_transitions, replace=True)
        
        expert_obs_sample = all_expert_obs[expert_indices]
        expert_actions_sample = all_expert_actions[expert_indices]
        expert_sa = self._prepare_batch(expert_obs_sample, expert_actions_sample)
        
        c_expert_for_expert = th.ones(expert_sa.size(0), dtype=th.long, device=self.device)
        c_agent_for_expert = th.zeros(expert_sa.size(0), dtype=th.long, device=self.device)

        loss_pos_expert = self.discriminator.compute_loss(expert_sa, c_expert_for_expert)
        loss_neg_expert = self.discriminator.compute_loss(expert_sa, c_agent_for_expert)
        prob_for_expert_data = th.exp(-loss_pos_expert) / (th.exp(-loss_pos_expert) + th.exp(-loss_neg_expert) + 1e-8)
        loss_expert = F.binary_cross_entropy(prob_for_expert_data, th.ones_like(prob_for_expert_data))

        c_expert_for_agent = th.ones(agent_sa.size(0), dtype=th.long, device=self.device)
        c_agent_for_agent = th.zeros(agent_sa.size(0), dtype=th.long, device=self.device)

        loss_pos_agent = self.discriminator.compute_loss(agent_sa, c_expert_for_agent)
        loss_neg_agent = self.discriminator.compute_loss(agent_sa, c_agent_for_agent)
        prob_for_agent_data = th.exp(-loss_pos_agent) / (th.exp(-loss_pos_agent) + th.exp(-loss_neg_agent) + 1e-8)
        loss_agent = F.binary_cross_entropy(prob_for_agent_data, th.zeros_like(prob_for_agent_data))

        discriminator_loss = loss_expert + loss_agent
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        
        self._disc_update_count += 1
        # self._disc_freeze_counter = 0
        
        return discriminator_loss.item(), prob_for_expert_data.mean().item(), prob_for_agent_data.mean().item()

    def _update_policy(self, batch):
        obs = th.as_tensor(batch['obs'], dtype=th.float32, device=self.device)
        actions = th.as_tensor(batch['actions'], dtype=th.long, device=self.device)
        masks = th.as_tensor(batch['masks'], dtype=th.float32, device=self.device)
        next_obs = th.as_tensor(batch['next_obs'], dtype=th.float32, device=self.device)
        imitation_rewards = th.as_tensor(batch['rewards'], dtype=th.float32, device=self.device)

        agent_ids = th.eye(self.args.n_agents, device=self.device)
        agent_ids = agent_ids.unsqueeze(0).unsqueeze(0).expand(obs.shape[0], obs.shape[1], -1, -1)
        
        policy_input = th.cat([obs, agent_ids], dim=-1)

        value_input = obs
        next_value_input = next_obs
        
        with th.no_grad():
            values = self.value(value_input.reshape(-1, self.obs_shape)).reshape(obs.shape[0], obs.shape[1], obs.shape[2], 1)
            next_values = self.value(next_value_input.reshape(-1, self.obs_shape)).reshape(next_obs.shape[0], next_obs.shape[1], next_obs.shape[2], 1)
            advantages = get_gae(self.args, imitation_rewards, values, next_values, masks)
            returns = advantages + values

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            logits_old = self.policy(policy_input.reshape(-1, self.policy_input_dim))
            dist_old = th.distributions.Categorical(logits=logits_old)
            log_probs_old = dist_old.log_prob(actions.reshape(-1))

        policy_loss_total = 0
        value_loss_total = 0
        entropy_loss_total = 0

        policy_input_reshaped = policy_input.reshape(-1, self.policy_input_dim)
        actions_reshaped = actions.reshape(-1, 1)
        returns_reshaped = returns.reshape(-1, 1)
        advantages_reshaped = advantages.reshape(-1, 1)
        log_probs_old_reshaped = log_probs_old.reshape(-1, 1)
        
        dataset = th.utils.data.TensorDataset(
            policy_input_reshaped, 
            actions_reshaped, 
            returns_reshaped, 
            advantages_reshaped,
            log_probs_old_reshaped
        )
        loader = th.utils.data.DataLoader(dataset, batch_size=self.args.ppo_batch_size, shuffle=True)

        for _ in range(self.args.ppo_epochs):
            for mini_batch in loader:
                obs_b, actions_b, returns_b, advantages_b, log_probs_old_b = mini_batch

                logits_new = self.policy(obs_b)
                dist_new = th.distributions.Categorical(logits=logits_new)
                log_probs_new = dist_new.log_prob(actions_b.squeeze(-1))
                entropy = dist_new.entropy().mean()

                ratio = th.exp(log_probs_new - log_probs_old_b.squeeze(-1))

                surr1 = ratio * advantages_b.squeeze(-1)
                surr2 = th.clamp(ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantages_b.squeeze(-1)
                policy_loss = -th.min(surr1, surr2).mean()

                value_obs_b = obs_b[:, :self.obs_shape]
                value_pred = self.value(value_obs_b)
                value_loss = ((value_pred - returns_b) ** 2).mean()

                bc_loss = self._compute_bc_loss()

                loss = (policy_loss
                        - self.args.entropy_coef * entropy
                        + self.args.value_loss_coef * value_loss
                        + self.bc_loss_weight * bc_loss)

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_norm_clip)
                th.nn.utils.clip_grad_norm_(self.value.parameters(), self.args.grad_norm_clip)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_loss_total += entropy.item()

        return {
            'policy_loss': policy_loss_total / (self.args.ppo_epochs * len(loader)),
            'value_loss': value_loss_total / (self.args.ppo_epochs * len(loader)),
            'entropy': entropy_loss_total / (self.args.ppo_epochs * len(loader)),
        }

    def train(self, t_env):
        """
        The main training loop performs one full training step on a single batch of data.
        """
        if len(self.expert_transitions) < self.args.batch_size_run:
            print(f"Insufficient expert data ({len(self.expert_transitions)}/{self.args.batch_size_run})，skip this training。")
            return None

        if self._bc_pretrain_steps > 0:
            self._bc_pretrain()
            self._bc_pretrain_steps = 0

        obs, avail_actions, actions, env_rewards, masks, next_obs = self.buffer.sample()

        batch = {
            'obs': obs,
            'avail_actions': avail_actions,
            'actions': actions,
            'rewards': env_rewards,
            'masks': masks,
            'next_obs': next_obs
        }
            
        agent_sa = self._prepare_batch(batch['obs'], batch['actions'])

        disc_loss, prob_expert, prob_agent = self._update_discriminator(agent_sa)

        gail_disc_loss = self._update_gail_discriminator(agent_sa)

        with th.no_grad():
            imitation_rewards = self._compute_rewards(batch['obs'], batch['actions'], env_rewards, t_env)
        
        batch['rewards'] = imitation_rewards

        policy_stats = self._update_policy(batch)
        
        self.update_count += 1
        self._update_bc_weight()

        return {
            'discriminator_loss': disc_loss,
            'gail_discriminator_loss': gail_disc_loss,
            'prob_expert': prob_expert,
            'prob_agent': prob_agent,
            'avg_imitation_reward': imitation_rewards.mean().item(),
            **policy_stats
        }

    def update_performance_monitoring(self, success_rate: float):
        self._success_ema = 0.9 * self._success_ema + 0.1 * success_rate

        if self._use_simplified_reward:
            if self._success_ema < 0.75:
                self._w_factor = min(self._w_factor * 1.02, 1.5)
            elif self._success_ema > 0.99:
                self._w_factor = max(self._w_factor * 0.998, 0.9)
        else:
            if self._success_ema < self._target_success - 0.05:
                self._w_factor = min(self._w_factor * 1.05, 2.0)
            elif self._success_ema > self._target_success + 0.05:
                self._w_factor = max(self._w_factor * 0.999, 0.9)

        self.training_stats['w_factor'] = self._w_factor

        self.performance_history.append(success_rate)
        if len(self.performance_history) > 30:
            self.performance_history.pop(0)

    def save_models(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        th.save(self.policy.state_dict(), f"{path}/policy_net.pth")
        th.save(self.value.state_dict(), f"{path}/value_net.pth")
        th.save(self.discriminator.state_dict(), f"{path}/discriminator.pth")

    def load_models(self, path):
        self.policy.load_state_dict(th.load(f"{path}/policy_net.pth"))
        self.value.load_state_dict(th.load(f"{path}/value_net.pth"))
        self.discriminator.load_state_dict(th.load(f"{path}/discriminator.pth"))

    def reset_training_stats(self):
        self.training_stats = {}

    def _update_gail_discriminator(self, agent_sa):
        if len(self.expert_transitions) < self.args.batch_size_run:
            return 0.0
            
        if not self.expert_transitions:
            return 0.0
        all_expert_obs = th.cat([t[0] for t in self.expert_transitions], dim=0)
        all_expert_actions = th.cat([t[1] for t in self.expert_transitions], dim=0)

        num_transitions = agent_sa.size(0)
        expert_indices = np.random.choice(all_expert_obs.shape[0], num_transitions, replace=True)
        
        expert_obs_sample = all_expert_obs[expert_indices]
        expert_actions_sample = all_expert_actions[expert_indices]

        expert_sa = self._prepare_batch(expert_obs_sample, expert_actions_sample)
        
        prob_expert = self.gail_discriminator(expert_sa)
        prob_agent = self.gail_discriminator(agent_sa)

        loss = -(th.log(prob_expert + 1e-8).mean() + th.log(1 - prob_agent + 1e-8).mean())

        self.gail_discriminator_optimizer.zero_grad()
        loss.backward()
        self.gail_discriminator_optimizer.step()

        return loss.item()

    def _drail_or_gail_reward(self, state_action: th.Tensor, t_env: int):
        use_gail = self.args.switch_to_gail_at_step != -1 and t_env >= self.args.switch_to_gail_at_step

        with th.set_grad_enabled(True):
            if use_gail:
                # ---------- GAIL ----------
                prob_agent = self.gail_discriminator(state_action).clamp(1e-8, 1 - 1e-8)
                loss = -th.log(1.0 - prob_agent).squeeze(-1)                               # (B,)
            else:
                # ---------- DRAIL ----------
                c_expert = th.ones(state_action.size(0), dtype=th.long, device=self.device)
                c_agent  = th.zeros_like(c_expert)

                loss_expert = self.discriminator.compute_loss(state_action, c_expert)
                loss_agent  = self.discriminator.compute_loss(state_action, c_agent)

                # D_φ(s,a) = exp(−L_exp) / [exp(−L_exp)+exp(−L_agent)]
                prob_exp = th.exp(-loss_expert)
                prob_ag  = th.exp(-loss_agent)
                d_phi = prob_exp / (prob_exp + prob_ag + 1e-8)

                loss = -(th.log(d_phi + 1e-8) - th.log(1.0 - d_phi + 1e-8))

        return loss.detach()

    @th.no_grad()
    def _compute_rewards(self, obs, actions, env_r, t_env):
        state_action = self._prepare_batch(obs, actions)
        
        use_gail = self.args.switch_to_gail_at_step != -1 and t_env >= self.args.switch_to_gail_at_step
        
        if use_gail:
            with th.set_grad_enabled(True):
                prob_agent = self.gail_discriminator(state_action).clamp(1e-8, 1 - 1e-8)
                base_rewards = -th.log(1.0 - prob_agent).squeeze(-1)
        else:
            if self._use_simplified_reward:
                with th.set_grad_enabled(True):
                    prob_agent = self.gail_discriminator(state_action).clamp(1e-8, 1 - 1e-8)
                    base_rewards = -th.log(1.0 - prob_agent).squeeze(-1)
            else:
                with th.set_grad_enabled(True):
                    c_expert = th.ones(state_action.size(0), dtype=th.long, device=self.device)
                    c_agent = th.zeros_like(c_expert)
                    
                    loss_expert = self.discriminator.compute_loss(state_action, c_expert)
                    loss_agent = self.discriminator.compute_loss(state_action, c_agent)
                    
                    # D_φ(s,a) = exp(−L_exp) / [exp(−L_exp)+exp(−L_agent)]
                    prob_exp = th.exp(-loss_expert)
                    prob_ag = th.exp(-loss_agent)
                    d_phi = prob_exp / (prob_exp + prob_ag + 1e-8)
                    
                    base_rewards = -th.log(1.0 - d_phi + 1e-8)
        
        if self._use_simplified_reward:
            anneal_progress = min(1.0, t_env / 80000.0)
            weight = 1.0 - 0.2 * anneal_progress
            final_rewards = base_rewards * weight
        else:
            anneal_progress = min(1.0, t_env / getattr(self.args, 'w_anneal_tau', 50000))
            current_imit_weight = getattr(self.args, 'imit_weight_start', 1.0) * (1 - anneal_progress) + getattr(self.args, 'imit_weight_end', 0.1) * anneal_progress
            dynamic_scale = current_imit_weight * self._w_factor
            
            progress_factor = min(1.0, self.update_count / 3000.0)
            progress_scale = 0.5 + progress_factor * 0.5
            
            final_rewards = base_rewards * dynamic_scale * progress_scale
        
        env_r_flat = env_r.to(final_rewards.device).reshape(-1)
        final_rewards = final_rewards + 0.02 * env_r_flat
        
        return final_rewards.reshape(obs.shape[0], obs.shape[1], obs.shape[2], 1)

    def _bc_pretrain(self):
        for _ in range(self._bc_pretrain_steps):
            idx = np.random.choice(len(self.expert_transitions))
            obs_e, act_e = self.expert_transitions[idx]          # (1, n_agents, obs_dim)
            obs_e = obs_e.to(self.device)
            act_e = act_e.to(self.device)

            agent_ids = th.eye(self.n_agents, device=self.device)   # (n_agents, n_agents) 与 obs_e 同为 2 维
            inp = th.cat([obs_e, agent_ids], dim=-1).reshape(-1, self.policy_input_dim)

            logits = self.policy(inp)
            loss = F.cross_entropy(logits, act_e.view(-1))
            self._bc_optimizer.zero_grad()
            loss.backward()
            self._bc_optimizer.step()
