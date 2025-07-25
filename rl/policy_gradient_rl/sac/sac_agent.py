"""
[2018 ICML SAC v1] Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
[2018 arXiv SAC v2] Soft Actor-Critic Algorithms and Applications
[2019 arXiv Discrete SAC]-Soft Actor-Critic for Discrete Action Settings
[2020 NIPS CQL] Conservative Q-Learning for Ofﬂine Reinforcement Learning
"""
from copy import deepcopy

import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from buffer.replay_buffer import ReplayBuffer
from rl.policy_gradient_rl.sac.sac_network import DQN, Actor, Critic
from utils.action_selectors import soft_selector, random_selector, greedy_selector
from utils.rl_utils import soft_update, hard_update
import numpy as np


class SACAgent:
    def __init__(self, args):
        self.args = args
        self.device = th.device(args.device)

        self.dqn1 = DQN(args).to(self.device)
        self.dqn2 = DQN(args).to(self.device)
        self.target_dqn1 = deepcopy(self.dqn1)
        self.target_dqn2 = deepcopy(self.dqn2)
        if args.add_critic:
            self.critic = Critic(args).to(self.device)
            self.critic_optimizer = Adam(self.critic.parameters(), args.lr)

        self.actor = Actor(args).to(self.device)

        self.dqn1_optimizer = Adam(params=self.dqn1.parameters(), lr=args.lr)
        self.dqn2_optimizer = Adam(params=self.dqn2.parameters(), lr=args.lr)

        self.actor_optimizer = Adam(params=self.actor.parameters(), lr=args.lr)
        if self.args.adaptive_alpha:
            self.log_alpha = th.zeros(1, dtype=th.float, requires_grad=True, device=self.device)
            self.target_entropy = -np.log(1.0 / args.n_actions) * 0.98
            self.alpha_optimizer = Adam(params=[self.log_alpha], lr=args.lr)
        self.buffer = ReplayBuffer(args)

    @th.no_grad()
    def select_actions(self, obs, avail_actions, test_mode=False):
        # trick 1 prevents SAC from getting trapped in a suboptimal solution
        if self.buffer.size < self.args.start_training_size and not test_mode:
            picked_actions = random_selector(avail_actions)
        else:
            obs = obs.to(self.device)
            avail_actions = avail_actions.to(self.device)
            agent_ids = th.eye(self.args.n_agents).to(self.device)
            obs = th.cat([obs, agent_ids], dim=-1)
            agent_outs = self.actor(obs)
            agent_outs[avail_actions == 0] = -1e10
            if test_mode:
                picked_actions = greedy_selector(agent_outs)
            else:
                picked_actions = soft_selector(agent_outs)
        return picked_actions

    def train(self):
        # trick 1 prevents SAC from getting trapped in a suboptimal solution
        if not self.args.offline and self.buffer.size < self.args.start_training_size:
            return
        obs, avail_actions, actions, rewards, masks, next_obs = self.buffer.sample()
        agent_ids = th.eye(self.args.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)
        obs = th.cat([obs, agent_ids], dim=-1)
        next_obs = th.cat([next_obs, agent_ids], dim=-1)

        if self.args.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        q1 = self.dqn1(obs)
        q2 = self.dqn2(obs)
        q1_a = q1.gather(-1, actions)
        q2_a = q2.gather(-1, actions)

        if self.args.adaptive_alpha:
            alpha = th.exp(self.log_alpha.detach())
        else:
            alpha = self.args.alpha
        with th.no_grad():
            next_log_prob = F.log_softmax(self.actor(next_obs), dim=-1)
            target_q = next_log_prob.exp() * (th.min(self.target_dqn1(next_obs), self.target_dqn2(next_obs)) - alpha * next_log_prob)
            E_target_q = target_q.sum(dim=-1, keepdim=True)
            target_q_v = rewards + masks * self.args.gamma * E_target_q

        q_v_loss1 = F.mse_loss(q1_a, target_q_v)
        q_v_loss2 = F.mse_loss(q2_a, target_q_v)

        if self.args.use_cql:
            cql_loss1 = th.logsumexp(q1, dim=-1).mean() - q1_a.mean()
            q_v_loss1 += self.args.cql_weight * cql_loss1
            cql_loss2 = th.logsumexp(q2, dim=-1).mean() - q2_a.mean()
            q_v_loss2 += self.args.cql_weight * cql_loss2

        self.dqn1_optimizer.zero_grad()
        q_v_loss1.backward()
        self.dqn1_optimizer.step()

        self.dqn2_optimizer.zero_grad()
        q_v_loss2.backward()
        self.dqn2_optimizer.step()

        log_prob = F.log_softmax(self.actor(obs), dim=-1)
        q_min = th.min(q1, q2)
        actor_loss = -(log_prob.exp() * (q_min.detach() - alpha * log_prob)).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        """-----------------------------------------"""
        if self.args.add_critic:
            value = self.critic(obs)
            critic_loss = F.mse_loss(value, target_q_v)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        """-----------------------------------------"""
        if self.args.adaptive_alpha:
            alpha_loss = -th.mean(self.log_alpha * (log_prob.detach() + self.target_entropy))
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        if self.args.soft_update:
            soft_update(self.args, self.target_dqn1, self.dqn1)
            soft_update(self.args, self.target_dqn2, self.dqn2)
        else:
            hard_update(self.target_dqn1, self.dqn1)
            hard_update(self.target_dqn2, self.dqn2)

    def save_models(self, path):
        th.save(self.dqn1.state_dict(), "{}/dqn1.th".format(path))
        th.save(self.dqn2.state_dict(), "{}/dqn2.th".format(path))
        th.save(self.actor.state_dict(), "{}/actor.th".format(path))
        if self.args.add_critic:
            th.save(self.critic.state_dict(), f"{path}/critic.th")
        if self.args.adaptive_alpha:
            th.save(self.log_alpha, f'{path}/alpha.th')

    def load_models(self, path):
        self.dqn1.load_state_dict(th.load(f'{path}/dqn1.th', map_location=lambda storage, loc: storage))
        self.dqn2.load_state_dict(th.load(f'{path}/dqn2.th', map_location=lambda storage, loc: storage))
        self.target_dqn1 = deepcopy(self.dqn1)
        self.target_dqn2 = deepcopy(self.dqn2)
        self.actor.load_state_dict(th.load(f'{path}/actor.th', map_location=lambda storage, loc: storage))
        if self.args.add_critic:
            self.critic.load_state_dict(th.load(f'{path}/critic.th', map_location=lambda storage, loc: storage))
        if self.args.adaptive_alpha:
            self.log_alpha = th.load(f'{path}/alpha.th')

    def load_offline_models(self, path):
        self.load_models(path)
