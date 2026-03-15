import torch 
import torch.nn as nn
from config import VexConfig

class BallEncoder(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        self.fc_1 = nn.Linear(config.ball_obs_dim, config.n_embd // 2)
        self.gelu = nn.GELU()
        self.fc_2 = nn.Linear(config.n_embd // 2, config.n_embd // 2)
        self.ln = nn.LayerNorm(config.n_embd // 2)

    def forward(self, ball_obs):
        # ball_obs: (B, block_size, n_balls, ball_obs_dim)
        x = self.gelu(self.fc_1(ball_obs))
        x = self.gelu(self.fc_2(x)) # (B, block_size, n_balls, n_embd//2)
        x = torch.max(x, dim=-2).values # (B, block_size, n_embd//2)
        x = self.ln(x)
        return x

class CoreEncoder(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        self.fc_1 = nn.Linear(config.core_obs_dim, config.n_embd // 2)
        self.gelu = nn.GELU()
        self.fc_2 = nn.Linear(config.n_embd // 2, config.n_embd // 2)
        self.ln = nn.LayerNorm(config.n_embd // 2)

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

        self.fc_1 = nn.Linear(config.n_embd, config.n_embd)
        self.gelu = nn.GELU()
        self.fc_2 = nn.Linear(config.n_embd, config.n_embd)
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

