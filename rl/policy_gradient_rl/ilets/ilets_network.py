import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, args):

        super(Actor, self).__init__()
        self.args = args
        
        input_dim = args.obs_shape + args.n_agents
        
        self.fc1 = nn.Linear(input_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc_out = nn.Linear(args.hidden_dim, args.n_actions)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.1)
        nn.init.constant_(self.fc_out.bias, 0.0)
    
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        action_logits = self.fc_out(x)
        
        return action_logits


class Critic(nn.Module):

    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        
        input_dim = args.obs_shape + args.n_agents
        
        self.fc1 = nn.Linear(input_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc_out = nn.Linear(args.hidden_dim, 1)  # 输出标量价值
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0.0)
    
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        value = self.fc_out(x)
        
        return value
