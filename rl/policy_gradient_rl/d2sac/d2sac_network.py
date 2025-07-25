from typing import Optional

import torch as th
from torch import nn
import torch.nn.functional as F


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_steps, embed_dim)

    def forward(self, t: th.Tensor) -> th.Tensor:
        return self.embedding(t)


class DiffusionActor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.T = args.diffusion_steps
        hidden = args.hidden_dim
        input_dim = args.obs_shape + args.n_agents
        self.state_encoder = nn.Linear(input_dim, hidden)
        self.time_embed = DiffusionEmbedding(self.T, hidden)
        self.denoise_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # decoder to logits
        self.decoder = nn.Linear(hidden, args.n_actions)

    def _denoise_step(self, h: th.Tensor, t: int) -> th.Tensor:
        # 加入时间嵌入
        t_emb = self.time_embed(th.tensor([t], device=h.device)).squeeze(0)
        out = h + t_emb
        out = self.denoise_net(out)
        return out

    def forward(self, inputs: th.Tensor) -> th.Tensor:
        # inputs: (batch, obs_dim)
        h = self.state_encoder(inputs)
        for t in reversed(range(self.T)):
            h = self._denoise_step(h, t)
        logits = self.decoder(h)
        return logits


class DQN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.obs_shape + args.n_agents, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class Critic(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.obs_shape + args.n_agents, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v