"""
[2017 arXiv PPO]-Proximal Policy Optimization Algorithms
"""
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from buffer.episode_buffer import EpisodeBuffer
from rl.policy_gradient_rl.ppo.ppo_network import Actor, Critic
from utils.action_selectors import soft_selector, greedy_selector
from utils.advantage_utils import get_gae


class PPOAgent:
    def __init__(self, args):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.args = args

        self.actor = Actor(args).to(self.device)
        self.critic = Critic(args).to(self.device)
        # trick
        if args.use_adm_epsilon:
            self.actor_optimizer = Adam(params=self.actor.parameters(), lr=args.lr, eps=args.adam_epsilon)
            self.critic_optimizer = Adam(params=self.critic.parameters(), lr=args.lr, eps=args.adam_epsilon)
        else:
            self.actor_optimizer = Adam(params=self.actor.parameters(), lr=args.lr)
            self.critic_optimizer = Adam(params=self.critic.parameters(), lr=args.lr)
        self.actor_lr_scheduler = th.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=args.lr_decay_step,
                                                               gamma=args.lr_decay_gamma)
        self.critic_lr_scheduler = th.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=args.lr_decay_step,
                                                                gamma=args.lr_decay_gamma)
        self.buffer = EpisodeBuffer(args)

    def train_gnn(self):
        """train GNN model"""
        # #1 ==================================
        if self.loss is not None:
            loss = self.loss
            loss = loss.clone().detach()

            # 计算 MSELoss
            mse_loss_fn = th.nn.MSELoss()
            loss_value = mse_loss_fn(loss, th.tensor(0.0, device=loss.device))  # 计算损失值
            # 确保损失值需要梯度
            loss_value = loss_value.requires_grad_()

            self.gcn_optimizer.zero_grad()
            loss_value.backward()  # 使用计算得到的损失值
            self.gcn_optimizer.step()

    def train_gnn2(self, obs, edge_index):
        # #2 ================================
        self.gcn_optimizer.zero_grad()
        gcn_loss = self.gcn_model.reconstruction_loss(obs, edge_index)  # obs为输入特征
        gcn_loss.backward()
        self.gcn_optimizer.step()


    @th.no_grad()
    def select_actions(self, obs, avail_actions, test_mode=False):
        test_mode = False
        obs = obs.to(self.device)
        agent_ids = th.eye(self.args.n_agents).unsqueeze(0).expand(self.args.n_threads, -1, -1).to(self.device)
        obs = th.cat([obs, agent_ids], dim=-1)
        avail_actions = avail_actions.to(self.device)
        actor_outs = self.actor(obs)
        # print("actor_outs:", actor_outs)
        actor_outs[avail_actions == 0] = -1e10
        if test_mode:
            picked_actions = greedy_selector(actor_outs)
        else:
            picked_actions = soft_selector(actor_outs)
        return picked_actions

    def evaluate_actions(self, inputs, avail_actions, actions):
        actor_outs = self.actor(inputs)
        actor_outs[avail_actions == 0] = -1e10
        probs = F.softmax(actor_outs, dim=-1)
        dist = Categorical(probs)
        log_probs_taken = dist.log_prob(actions.squeeze(-1))
        entropy = dist.entropy()
        return log_probs_taken.unsqueeze(-1), entropy.unsqueeze(-1)

    def train(self):
        obs, avail_actions, actions, rewards, masks, next_obs = self.buffer.sample()

        agent_ids = th.eye(self.args.n_agents).unsqueeze(0).unsqueeze(0).expand(self.args.n_threads, self.args.episode_limit, -1, -1).to(self.device)
        obs = th.cat([obs, agent_ids], dim=-1)
        next_obs = th.cat([next_obs, agent_ids], dim=-1)

        # trick
        if self.args.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_log_pi_taken, _ = self.evaluate_actions(obs, avail_actions, actions)

        with th.no_grad():
            old_v = self.critic(obs)
            old_next_v = self.critic(next_obs)
            advantages = get_gae(self.args, rewards, old_v, old_next_v, masks)
        # trick no work
        if self.args.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        targets = advantages + old_v
        for k in range(self.args.epochs):
            log_pi_taken, entropy = self.evaluate_actions(obs, avail_actions, actions)
            v = self.critic(obs)
            critic_loss = (((v - targets) * masks) ** 2).sum() / masks.sum()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # trick
            if self.args.use_grad_clip:
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_norm_clip)
            self.critic_optimizer.step()
            self.critic_lr_scheduler.step()

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages
            if self.args.use_entropy:
                actor_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * masks).sum() / masks.sum()
            else:
                actor_loss = -((th.min(surr1, surr2)) * masks).sum() / masks.sum()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # trick
            if self.args.use_grad_clip:
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
            self.actor_optimizer.step()
            self.actor_lr_scheduler.step()

    def save_models(self, path):
        th.save(self.actor.state_dict(), "{}/actor.th".format(path))
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))

    def load_models(self, path):
        self.actor.load_state_dict(th.load("{}/actor.th".format(path), map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))

    def load_offline_models(self, path):
        self.load_models(path)
