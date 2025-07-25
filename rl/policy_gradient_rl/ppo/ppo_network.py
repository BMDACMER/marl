import torch.nn as nn

from utils.rl_utils import orthogonal_init, activate_funs


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.obs_shape + args.n_agents, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)
        # trick
        self.activate_fun = activate_funs[args.activate_fun]
        # trick
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, inputs):
        x = self.activate_fun(self.fc1(inputs))
        x = self.activate_fun(self.fc2(x))
        q = self.fc3(x)
        return q


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.obs_shape + args.n_agents, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

        self.activate_fun = activate_funs[args.activate_fun]

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, inputs):
        x = self.activate_fun(self.fc1(inputs))
        x = self.activate_fun(self.fc2(x))
        q = self.fc3(x)
        return q
