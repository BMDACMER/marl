import torch as th


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.obs = th.zeros(args.buffer_size, args.episode_limit, args.n_agents, args.obs_shape, dtype=th.float)
        self.avail_actions = th.zeros(args.buffer_size, args.episode_limit, args.n_agents, args.n_actions, dtype=th.int)
        self.actions = th.zeros(args.buffer_size, args.episode_limit, args.n_agents, 1, dtype=th.long)
        self.rewards = th.zeros(args.buffer_size, args.episode_limit, args.n_agents, 1, dtype=th.float)
        self.masks = th.zeros(args.buffer_size, args.episode_limit, args.n_agents, 1, dtype=th.float)
        self.next_obs = th.zeros(args.buffer_size, args.episode_limit, args.n_agents, args.obs_shape, dtype=th.float)
        self.size = 0
        self.index = 0
        self.device = th.device(args.device)

    def insert(self, episode_buffer):
        for i in range(self.args.n_threads):
            self.obs[self.index] = episode_buffer.obs[i].clone()
            self.avail_actions[self.index] = episode_buffer.avail_actions[i].clone()
            self.actions[self.index] = episode_buffer.actions[i].clone()
            self.rewards[self.index] = episode_buffer.rewards[i].clone()
            self.masks[self.index] = episode_buffer.masks[i].clone()
            self.next_obs[self.index] = episode_buffer.next_obs[i].clone()

            self.index = (1 + self.index) % self.args.buffer_size
            self.size = min(self.args.buffer_size, self.size + 1)

    def sample(self):
        perm = th.randperm(self.size)
        batch = perm[:self.args.batch_size]
        obs = self.obs[batch].to(self.device)
        avail_actions = self.avail_actions[batch].to(self.device)
        actions = self.actions[batch].to(self.device)
        rewards = self.rewards[batch].to(self.device)
        masks = self.masks[batch].to(self.device)
        next_obs = self.next_obs[batch].to(self.device)
        return obs, avail_actions, actions, rewards, masks, next_obs

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
