import numpy as np
import torch as th


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.obs = th.zeros(args.buffer_size, args.n_agents, args.obs_shape, dtype=th.float)
        self.avail_actions = th.zeros(args.buffer_size, args.n_agents, args.n_actions, dtype=th.int)
        self.actions = th.zeros(args.buffer_size, args.n_agents, 1, dtype=th.long)
        self.rewards = th.zeros(args.buffer_size, args.n_agents, 1, dtype=th.float)
        self.masks = th.zeros(args.buffer_size, args.n_agents, 1, dtype=th.float)
        self.next_obs = th.zeros(args.buffer_size, args.n_agents, args.obs_shape, dtype=th.float)
        self.size = 0
        self.index = 0
        self.device = th.device(args.device)

    def insert(self, obs, avail_actions, actions, rewards, masks, next_obs):
        self.obs[self.index] = obs
        self.avail_actions[self.index] = avail_actions
        self.actions[self.index] = actions
        self.rewards[self.index] = rewards
        self.masks[self.index] = masks
        self.next_obs[self.index] = next_obs

        self.index = (1 + self.index) % self.args.buffer_size
        self.size = min(self.args.buffer_size, self.size + 1)

    def sample(self):
        # perm = th.randperm(self.size)
        # batch = perm[:self.args.batch_size]
        batch = np.random.choice(self.size, size=self.args.batch_size)
        obs = self.obs[batch].to(self.device)
        avail_actions = self.avail_actions[batch].to(self.device)
        actions = self.actions[batch].to(self.device)
        rewards = self.rewards[batch].to(self.device)
        masks = self.masks[batch].to(self.device)
        next_obs = self.next_obs[batch].to(self.device)
        return obs, avail_actions, actions, rewards, masks, next_obs

    # def n_step_sample(self):
    #     obs = th.zeros(self.args.batch_size, self.args.n_agents, self.args.obs_shape, dtype=th.float)
    #     avail_actions = th.zeros(self.args.batch_size, self.args.n_agents, self.args.n_actions, dtype=th.int)
    #     actions = th.zeros(self.args.batch_size, self.args.n_agents, 1, dtype=th.long)
    #     rewards = th.zeros(self.args.batch_size, self.args.n_agents, 1, dtype=th.float)
    #     masks = th.zeros(self.args.batch_size, self.args.n_agents, 1, dtype=th.float)
    #     next_obs = th.zeros(self.args.batch_size, self.args.n_agents, self.args.obs_shape, dtype=th.float)
    #
    #     for i in range(self.args.batch_size):
    #         # Will not reach end
    #         end = random.randint(self.args.n_step, self.size)
    #         start = end - self.args.n_step
    #         obs[i] = self.obs[start]
    #         avail_actions[i] = self.avail_actions[start]
    #         actions[i] = self.actions[start]
    #         for j in range(self.args.n_step):
    #             rewards[i] += (self.args.gamma ** j) * self.rewards[start + j]
    #             if self.masks[start + j][0] == 0.0:
    #                 masks[i] = self.masks[start + j]
    #                 next_obs[i] = self.next_obs[start + j]
    #                 break
    #         masks[i] = self.masks[end - 1]
    #         next_obs[i] = self.next_obs[end - 1]
    #     return obs.to(self.device), avail_actions.to(self.device), actions.to(self.device), rewards.to(self.device), masks.to(self.device), next_obs.to(self.device)

    def save(self, path):
        th.save(self.obs, f'{path}/obs.th')
        th.save(self.avail_actions, f'{path}/avail_actions.th')
        th.save(self.actions, f'{path}/actions.th')
        th.save(self.rewards, f'{path}/rewards.th')
        th.save(self.masks, f'{path}/masks.th')
        th.save(self.next_obs, f'{path}/next_obs.th')

    def load(self, path):
        self.obs = th.load(f'{path}/obs.th')
        self.avail_actions = th.load(f'{path}/avail_actions.th')
        self.actions = th.load(f'{path}/actions.th')
        self.rewards = th.load(f'{path}/rewards.th')
        self.masks = th.load(f'{path}/masks.th')
        self.next_obs = th.load(f'{path}/next_obs.th')
        self.size = self.obs.size(0)
        self.index = self.size - 1

    def normalize_rewards(self, eps=1e-3):
        rewards_mean = self.rewards.mean(dim=(0, 1), keepdim=True)
        rewards_std = self.rewards.std(dim=(0, 1), keepdim=True) + eps
        self.rewards = (self.rewards - rewards_mean) / rewards_std

    # def normalize_obs(self, eps=1e-3):
    #     obs_mean = self.obs.mean(dim=(0, 1), keepdim=True)
    #     obs_std = self.obs.std(dim=(0, 1), keepdim=True) + eps
    #     self.obs = (self.obs - obs_mean) / obs_std
    #     self.next_obs = (self.next_obs - obs_mean) / obs_std
