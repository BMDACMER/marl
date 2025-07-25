import torch as th


class PPOExpertBuffer:
    def __init__(self, args):
        self.args = args
        self.obs = th.zeros(args.n_threads, args.episode_limit, args.n_agents, args.obs_shape, dtype=th.float)
        self.avail_actions = th.zeros(args.n_threads, args.episode_limit, args.n_agents, args.n_actions, dtype=th.int)
        self.actions = th.zeros(args.n_threads, args.episode_limit, args.n_agents, 1, dtype=th.long)
        # self.actions_expert = th.zeros(args.n_threads, args.episode_limit, args.n_agents, 1, dtype=th.long)
        self.rewards = th.zeros(args.n_threads, args.episode_limit, args.n_agents, 1, dtype=th.float)
        self.masks = th.zeros(args.n_threads, args.episode_limit, args.n_agents, 1, dtype=th.float)
        self.next_obs = th.zeros(args.n_threads, args.episode_limit, args.n_agents, args.obs_shape, dtype=th.float)
        self.t = 0
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    def insert(self, obs, avail_actions, actions, rewards, masks, next_obs):
        self.obs[:, self.t] = obs
        self.avail_actions[:, self.t] = avail_actions
        self.actions[:, self.t] = actions
        self.rewards[:, self.t] = rewards
        self.masks[:, self.t] = masks
        self.next_obs[:, self.t] = next_obs
        self.t += 1

    def insertAll(self, obs, avail_actions, actions, rewards, masks, next_obs):
        self.obs[:] = obs
        self.avail_actions[:] = avail_actions
        self.actions[:] = actions
        self.rewards[:] = rewards
        self.masks[:] = masks
        self.next_obs[:] = next_obs

    def sample(self):
        obs = self.obs.to(self.device)
        avail_actions = self.avail_actions.to(self.device)
        actions = self.actions.to(self.device)
        rewards = self.rewards.to(self.device)
        masks = self.masks.to(self.device)
        next_obs = self.next_obs.to(self.device)
        return obs, avail_actions, actions, rewards, masks, next_obs

    def reset(self):
        self.obs.zero_()
        self.avail_actions.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.masks.zero_()
        self.next_obs.zero_()
        self.t = 0
