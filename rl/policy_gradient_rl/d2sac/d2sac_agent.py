from copy import deepcopy
from typing import Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from buffer.replay_buffer import ReplayBuffer
from rl.policy_gradient_rl.d2sac.d2sac_network import DQN, Critic, DiffusionActor
from utils.action_selectors import soft_selector, random_selector
from utils.rl_utils import soft_update, hard_update


class D2SACAgent:
    def __init__(self, args):
        self.args = args
        self.device = th.device(args.device)

        self.dqn1 = DQN(args).to(self.device)
        self.dqn2 = DQN(args).to(self.device)
        self.target_dqn1 = deepcopy(self.dqn1)
        self.target_dqn2 = deepcopy(self.dqn2)

        if args.add_critic:
            self.critic = Critic(args).to(self.device)
            self.critic_opt = Adam(self.critic.parameters(), args.lr)

        self.actor = DiffusionActor(args).to(self.device)

        # Optimizers
        self.dqn1_opt = Adam(self.dqn1.parameters(), args.lr)
        self.dqn2_opt = Adam(self.dqn2.parameters(), args.lr)
        self.actor_opt = Adam(self.actor.parameters(), args.lr)

        # alpha (entropy temperature)
        if self.args.adaptive_alpha:
            self.log_alpha = th.zeros(1, dtype=th.float, requires_grad=True, device=self.device)
            self.target_entropy = -np.log(1.0 / args.n_actions) * 0.98
            self.alpha_opt = Adam([self.log_alpha], args.lr)
        else:
            self.log_alpha = None

        self.buffer = ReplayBuffer(args)
        if args.expert_buffers_path:
            self._load_expert_data(args.expert_buffers_path)

    # ------------------------------------------------------------------
    @th.no_grad()
    def select_actions(self, obs, avail_actions, test_mode: bool = False):
        if self.buffer.size < self.args.start_training_size and not test_mode:
            return random_selector(avail_actions)

        obs = obs.to(self.device)
        avail_actions = avail_actions.to(self.device)
        agent_ids = th.eye(self.args.n_agents, device=self.device)
        obs = th.cat([obs, agent_ids], dim=-1)
        logits = self.actor(obs)
        logits[avail_actions == 0] = -1e9
        return soft_selector(logits)

    def _prepare_batch(self):
        obs, avail_actions, actions, rewards, masks, next_obs = self.buffer.sample()
        agent_ids = th.eye(self.args.n_agents, device=self.device).unsqueeze(0).expand(self.args.batch_size, -1, -1)
        obs = th.cat([obs, agent_ids], dim=-1)
        next_obs = th.cat([next_obs, agent_ids], dim=-1)
        if self.args.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "next_obs": next_obs,
        }

    def _get_alpha(self):
        if self.args.adaptive_alpha:
            return th.exp(self.log_alpha.detach())
        return self.args.alpha

    def _compute_target_q(self, batch, alpha):
        with th.no_grad():
            next_logits = self.actor(batch["next_obs"])
            next_log_prob = F.log_softmax(next_logits, dim=-1)
            min_next_q = th.min(self.target_dqn1(batch["next_obs"]), self.target_dqn2(batch["next_obs"]))
            target_q = (next_log_prob.exp() * (min_next_q - alpha * next_log_prob)).sum(-1, keepdim=True)
            target_q = batch["rewards"] + batch["masks"] * self.args.gamma * target_q
        return target_q

    def _update_q(self, batch, target_q):
        q1 = self.dqn1(batch["obs"]).gather(-1, batch["actions"])
        q2 = self.dqn2(batch["obs"]).gather(-1, batch["actions"])
        loss_q1 = F.mse_loss(q1, target_q)
        loss_q2 = F.mse_loss(q2, target_q)
        if self.args.use_cql:
            cql1 = th.logsumexp(self.dqn1(batch["obs"]), dim=-1).mean() - q1.mean()
            cql2 = th.logsumexp(self.dqn2(batch["obs"]), dim=-1).mean() - q2.mean()
            loss_q1 += self.args.cql_weight * cql1
            loss_q2 += self.args.cql_weight * cql2
        self._optimize(self.dqn1_opt, loss_q1, self.dqn1)
        self._optimize(self.dqn2_opt, loss_q2, self.dqn2)

    def _update_actor(self, batch, alpha):
        logits = self.actor(batch["obs"])
        log_prob = F.log_softmax(logits, dim=-1)
        q_min = th.min(self.dqn1(batch["obs"]), self.dqn2(batch["obs"]))
        loss_actor = -(log_prob.exp() * (q_min.detach() - alpha * log_prob)).sum(-1).mean()
        self._optimize(self.actor_opt, loss_actor, self.actor)
        return log_prob.detach()

    def _update_critic(self, batch, target_q):
        if not self.args.add_critic:
            return
        value = self.critic(batch["obs"])
        loss_value = F.mse_loss(value, target_q)
        self._optimize(self.critic_opt, loss_value, self.critic)

    def _update_alpha(self, log_prob):
        if not self.args.adaptive_alpha:
            return
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self._optimize(self.alpha_opt, alpha_loss, [self.log_alpha])

    def _optimize(self, optimizer, loss, params):
        optimizer.zero_grad()
        loss.backward()

        if self.args.grad_clip > 0:
            if isinstance(params, th.nn.Module):
                clip_params = params.parameters()
            else:
                clip_params = params
            th.nn.utils.clip_grad_norm_(clip_params, self.args.grad_clip)

        optimizer.step()

    def _sync_targets(self):
        if self.args.soft_update:
            soft_update(self.args, self.target_dqn1, self.dqn1)
            soft_update(self.args, self.target_dqn2, self.dqn2)
        else:
            hard_update(self.target_dqn1, self.dqn1)
            hard_update(self.target_dqn2, self.dqn2)

    def train(self):
        if self.buffer.size < self.args.batch_size:
            return
        batch = self._prepare_batch()
        alpha = self._get_alpha()
        target_q = self._compute_target_q(batch, alpha)
        self._update_q(batch, target_q)
        log_prob = self._update_actor(batch, alpha)
        self._update_critic(batch, target_q)
        self._update_alpha(log_prob)
        self._sync_targets()

    # ------------------------------------------------------------------
    def save_models(self, path: str):
        th.save(self.dqn1.state_dict(), f"{path}/d2sac_dqn1.th")
        th.save(self.dqn2.state_dict(), f"{path}/d2sac_dqn2.th")
        th.save(self.actor.state_dict(), f"{path}/d2sac_actor.th")
        if self.args.add_critic:
            th.save(self.critic.state_dict(), f"{path}/d2sac_critic.th")
        if self.args.adaptive_alpha:
            th.save(self.log_alpha, f"{path}/d2sac_alpha.th")

    def load_models(self, path: str):
        self.dqn1.load_state_dict(th.load(f"{path}/d2sac_dqn1.th", map_location=self.device))
        self.dqn2.load_state_dict(th.load(f"{path}/d2sac_dqn2.th", map_location=self.device))
        self.actor.load_state_dict(th.load(f"{path}/d2sac_actor.th", map_location=self.device))
        self.target_dqn1 = deepcopy(self.dqn1)
        self.target_dqn2 = deepcopy(self.dqn2)
        if self.args.add_critic:
            self.critic.load_state_dict(th.load(f"{path}/d2sac_critic.th", map_location=self.device))
        if self.args.adaptive_alpha:
            self.log_alpha = th.load(f"{path}/d2sac_alpha.th")

    # ------------------------------------------------------------------
    def _load_expert_data(self, path: str):
        try:
            data = th.load(path, map_location='cpu')  # 预期为 List[dict]
            for traj in data:
                self.buffer.insert(**traj)  # 假设字典字段与 ReplayBuffer.insert 对齐
            print(f"[D2SAC] Expert data loaded: {len(data)} transitions")
        except FileNotFoundError:
            print(f"[D2SAC]Expert data file not found : {path}, it will be trained from scratch")
        except Exception as e:
            print(f"[D2SAC] Error loading expert data: {e}")
