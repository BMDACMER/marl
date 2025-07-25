import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):

    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        
        input_dim = args.obs_shape + args.n_agents
        
        self.fc1 = nn.Linear(input_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)
        
        if getattr(args, 'activate_fun', 'relu') == 'relu':
            self.activate_func = F.relu
        elif args.activate_fun == 'tanh':
            self.activate_func = F.tanh
        else:
            self.activate_func = F.relu

    def forward(self, obs):
        x = self.activate_func(self.fc1(obs))
        x = self.activate_func(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic(nn.Module):

    def __init__(self, args):

        super(Critic, self).__init__()
        self.args = args
        
        input_dim = args.obs_shape + args.n_agents
        
        self.fc1 = nn.Linear(input_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        
        if getattr(args, 'activate_fun', 'relu') == 'relu':
            self.activate_func = F.relu
        elif args.activate_fun == 'tanh':
            self.activate_func = F.tanh
        else:
            self.activate_func = F.relu

    def forward(self, obs):
        x = self.activate_func(self.fc1(obs))
        x = self.activate_func(self.fc2(x))
        return self.fc3(x)


class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        
        input_dim = args.obs_shape + args.n_agents + args.n_actions
        
        self.fc1 = nn.Linear(input_dim, args.hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(args.hidden_dim * 2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.bn2 = nn.BatchNorm1d(args.hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(args.hidden_dim, args.hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(args.hidden_dim // 2)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(args.hidden_dim // 2, 1)
        
        if getattr(args, 'activate_fun', 'relu') == 'relu':
            self.activate_func = F.relu
        elif args.activate_fun == 'tanh':
            self.activate_func = F.tanh
        elif args.activate_fun == 'leaky_relu':
            self.activate_func = F.leaky_relu
        else:
            self.activate_func = F.relu
            
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, obs, actions):
        original_shape = obs.shape
        if len(original_shape) > 2:
            # 展平除最后一维外的所有维度
            obs = obs.view(-1, original_shape[-1])
            actions = actions.view(-1, actions.shape[-1])
        
        x = th.cat([obs, actions], dim=-1)
        
        x = self.fc1(x)
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = self.activate_func(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = self.activate_func(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        if x.shape[0] > 1:
            x = self.bn3(x)
        x = self.activate_func(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        # Restoring the original shape
        if len(original_shape) > 2:
            x = x.view(*original_shape[:-1], 1)
        
        return x
