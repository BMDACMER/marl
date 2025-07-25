import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits


class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)


class GAILDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GAILDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出一个概率值
        )

    def forward(self, x):
        return self.model(x)


class DiffusionDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden_dim

        self.time_embed = nn.Embedding(2, hidden_dim)          # t ∈ {0,1}
        self.cond_embed = nn.Embedding(2, hidden_dim)          # c ∈ {0,1}

        mlp_in = input_dim + 2 * hidden_dim
        self.mlp = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(mlp_in, hidden_dim)),
            nn.ReLU(),
            nn.utils.parametrizations.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)   # Prediction noise ε
        )

    def forward(self, x, t, c):
        """
        x : (B, input_dim)
        t : (B,)  0/1
        c : (B,)  0=agent / 1=expert
        """
        emb = th.cat([self.time_embed(t), self.cond_embed(c)], dim=-1)
        x_in = th.cat([x, emb], dim=-1)
        return self.mlp(x_in)

    # === Two-step backdiffusion loss ===
    def compute_loss(self, sa, c):
        """
        sa : (B, input_dim)
        c  : (B,)
        """
        B = sa.size(0)

        # Sampling random in two steps t ∈ {0,1}
        t = th.randint(0, 2, (B,), device=sa.device)
        noise = th.randn_like(sa)

        # q(x_t | x_0)   Sample the noisy input
        x_t = sa + (t.float().unsqueeze(-1) * noise)
        x_t.requires_grad_(True)
        
        # Prediction noise
        eps_pred = self.forward(x_t, t, c)

        # MSE
        mse = F.mse_loss(eps_pred, noise, reduction="none").mean(dim=1)

        # ----- R1 gradient penalty -----
        grad = th.autograd.grad(outputs=eps_pred.sum(), inputs=x_t,
                                create_graph=True, retain_graph=True)[0]
        r1 = grad.pow(2).view(B, -1).sum(1) * 1e-4       # λ_r1

        return mse + r1
