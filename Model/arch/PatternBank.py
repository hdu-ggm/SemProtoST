import torch
import torch.nn as nn
import torch.nn.functional as F

class PatternRelationBank(nn.Module):

    def __init__(self, num_prototypes, embed_dim):
        super().__init__()
        self.K = num_prototypes
        self.D = embed_dim

        # unified prototype memory
        self.bank_memory = nn.Parameter(torch.randn(self.K, self.D))
        nn.init.xavier_uniform_(self.bank_memory)

        # projections
        self.q_proj = nn.Conv2d(self.D, self.D, kernel_size=1)
        self.k_proj = nn.Linear(self.D, self.D)
        self.v_proj = nn.Linear(self.D, self.D)

        self.out_proj = nn.Conv2d(self.D, self.D, kernel_size=1)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, features, return_loss=True):
        """
        features: [B, D, N, 1]
        """
        B, D, N, _ = features.shape

        # Q
        q = self.q_proj(features).squeeze(-1).permute(0, 2, 1)  # [B, N, D]

        # K, V
        k = self.k_proj(self.bank_memory)  # [K, D]
        v = self.v_proj(self.bank_memory)  # [K, D]

        # attention
        attn = torch.matmul(q, k.T) / (D ** 0.5)  # [B, N, K]
        attn = torch.softmax(attn, dim=-1)

        # context
        context = torch.matmul(attn, v)  # [B, N, D]
        context = context.permute(0, 2, 1).unsqueeze(-1)  # [B, D, N, 1]

        out = features + self.out_proj(self.drop(self.act(context)))

        # ortho loss（弱约束）
        ortho_loss = torch.tensor(0.0, device=features.device)
        if return_loss:
            m = F.normalize(self.bank_memory, dim=1)
            sim = torch.matmul(m, m.T)
            ortho_loss = ((sim - torch.eye(self.K, device=features.device)) ** 2).mean()

        return out, attn, ortho_loss