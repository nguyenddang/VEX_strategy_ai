import torch 
import torch.nn as nn
from config import VexConfig

class BallEncoder(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        # attention over balls .
        self.pre_process = nn.Linear(config.ball_obs_dim, config.n_embd)
        self.ln_in = nn.LayerNorm(config.n_embd)
        self.qkv = nn.Linear(config.n_embd, config.n_embd * 3)
        self.attn_out = nn.Linear(config.n_embd, config.n_embd)
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.n_head = 4
        self.n_embd = config.n_embd

    def forward(self, ball_obs):
        B, N, _ = ball_obs.shape
        x = self.ln_in(self.pre_process(ball_obs)) # (B, block_size, n_balls, n_embd)
        q, k, v = torch.chunk(self.qkv(x), 3, dim=-1) # each is (B, block_size, n_balls, n_embd)
        q = q.view(B, N, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, n_head, n_balls, n_embd//n_head)
        k = k.view(B, N, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, n_head, n_balls, n_embd//n_head)
        v = v.view(B, N, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, n_head, n_balls, n_embd//n_head)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False) # (B, n_head, n_balls, n_embd//n_head)
        y = y.transpose(1, 2).contiguous().view(B, N, self.n_embd) # (B, n_balls, n_embd)
        y = torch.mean(y, dim=1) # (B, n_embd)
        y = self.ln_out(self.attn_out(y)) # (B, n_balls, n_embd)
        return y

class CoreEncoder(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        self.fc_1 = nn.Linear(config.core_obs_dim, config.n_embd)
        self.gelu = nn.GELU()
        self.fc_2 = nn.Linear(config.n_embd, config.n_embd)
        self.ln = nn.LayerNorm(config.n_embd)

    def forward(self, core_obs):
        # core_obs: (B, block_size, core_obs_dim)
        x = self.gelu(self.fc_1(core_obs))
        x = self.fc_2(x) # (B, block_size, n_embd//2)
        x = self.ln(x)
        return x

class Encoder(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        self.core_encoder = CoreEncoder(config)
        self.ball_encoder = BallEncoder(config)

        self.fc_1 = nn.Linear(config.n_embd * 2, config.n_embd * 2)
        self.gelu = nn.GELU()
        self.fc_2 = nn.Linear(config.n_embd * 2, config.n_embd)
        self.ln = nn.LayerNorm(config.n_embd)
    
    def forward(self, core_obs, ball_obs):
        # core_obs: (B, core_obs_dim)
        # ball_obs: (B, n_balls, ball_obs_dim)
        core_embed = self.core_encoder(core_obs) # (B, block_size, n_embd//2)
        ball_embed = self.ball_encoder(ball_obs) # (B, block_size, n_embd//2)
        x = torch.cat([core_embed, ball_embed], dim=-1) # (B, block_size, n_embd)

        x = self.gelu(self.fc_1(x))
        x = self.fc_2(x)
        x = self.ln(x)
        return x

