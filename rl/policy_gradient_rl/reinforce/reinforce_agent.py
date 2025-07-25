"""
[2017 arXiv PPO]-Proximal Policy Optimization Algorithms
"""
import torch as th
from rl.policy_gradient_rl.reinforce.reinforce_network import Actor
from torch.optim import Adam

from buffer.episode_buffer import EpisodeBuffer
from utils.action_selectors import soft_selector, greedy_selector
from utils.advantage_utils import get_returns


class ReinforceAgent:
    def __init__(self, args):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.args = args

        self.actor = Actor(args).to(self.device)
        self.actor_optimizer = Adam(params=self.actor.parameters(), lr=args.lr)

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

    # def evaluate_actions(self, inputs, avail_actions, actions):
    #     actor_outs = self.actor(inputs)
    #     actor_outs[avail_actions == 0] = -1e10
    #     probs = F.softmax(actor_outs, dim=-1)
    #     dist = Categorical(probs)
    #     log_probs_taken = dist.log_prob(actions.squeeze(-1))
    #     return log_probs_taken.unsqueeze(-1)

    def train(self):
        obs, avail_actions, actions, rewards, masks, next_obs = self.buffer.sample()
        agent_ids = th.eye(self.args.n_agents).unsqueeze(0).unsqueeze(0).expand(self.args.n_threads, self.args.episode_limit, -1, -1).to(self.device)
        obs = th.cat([obs, agent_ids], dim=-1)

        if self.args.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        returns = get_returns(self.args, rewards, masks)
        log_probs = th.log_softmax(self.actor(obs), dim=-1)
        log_probs_taken = log_probs.gather(-1, actions)

        actor_loss = - (log_probs_taken * returns).mean()
        # Optimise agents
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save_models(self, path):
        th.save(self.actor.state_dict(), "{}/actor.th".format(path))

    def load_models(self, path):
        self.actor.load_state_dict(th.load("{}/actor.th".format(path), map_location=lambda storage, loc: storage))
