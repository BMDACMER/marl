"""
"""
import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from buffer.episode_buffer import EpisodeBuffer
from rl.policy_gradient_rl.ac.ac_network import Actor, Critic
from utils.action_selectors import soft_selector, greedy_selector
from utils.advantage_utils import get_returns


class ActorCriticAgent:
    def __init__(self, args):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.args = args

        self.actor = Actor(args).to(self.device)
        self.actor_optimizer = Adam(params=self.actor.parameters(), lr=args.lr)

        self.critic = Critic(args).to(self.device)
        self.critic_optimizer = Adam(params=self.critic.parameters(), lr=args.lr)
        self.buffer = EpisodeBuffer(args)

    @th.no_grad()
    def select_actions(self, obs, avail_actions, test_mode=False):
        test_mode = False
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

    def evaluate_actions(self, inputs, avail_actions, actions):
        actor_outs = self.actor(inputs)
        actor_outs[avail_actions == 0] = -1e10
        log_probs = F.log_softmax(actor_outs, dim=-1)
        log_probs_taken = log_probs.gather(-1, actions)
        return log_probs_taken

    def train(self):
        obs, avail_actions, actions, rewards, masks, next_obs = self.buffer.sample()
        agent_ids = th.eye(self.args.n_agents).unsqueeze(0).unsqueeze(0).expand(self.args.n_threads, self.args.episode_limit, -1, -1).to(self.device)
        obs = th.cat([obs, agent_ids], dim=-1)

        if self.args.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        returns = get_returns(self.args, rewards, masks)

        v = self.critic(obs)
        advantages = (returns - v).detach()
        log_probs_taken = self.evaluate_actions(obs, avail_actions, actions)

        actor_loss = - (log_probs_taken * advantages * masks).sum() / masks.sum()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic_loss = (((v - returns) * masks) ** 2).sum() / masks.sum()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save_models(self, path):
        th.save(self.actor.state_dict(), "{}/actor.th".format(path))
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))

    def load_models(self, path):
        self.actor.load_state_dict(th.load("{}/actor.th".format(path), map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
