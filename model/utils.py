import torch 
import torch.nn as nn
from config import VexConfig


class BallEncoder(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        assert config.ndim % 2 == 0
        self.fc1 = nn.Linear(config.ball_obs_dim, config.ndim // 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.ndim // 2, config.ndim // 2)
        self.out_ln = nn.LayerNorm(config.ndim // 2)

    def forward(self, x):
        # x: (B * 2 * T, n_ball, ball_obs_dim)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = torch.max(x, dim=1).values # (B * 2 * T, ndim//2)
        x = self.out_ln(x)
        return x
    
class CoreEncoder(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        assert config.ndim % 2 == 0
        self.fc1 = nn.Linear(config.core_obs_dim, config.ndim // 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.ndim // 2, config.ndim // 2)
        self.out_ln = nn.LayerNorm(config.ndim // 2)

    def forward(self, x):
        # x: (B * 2 * T, context_window, core_obs_dim)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x)) # (B * 2 * T, context_window, ndim//2)
        x = self.out_ln(x)
        return x
    
class BasicEncoder(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.ndim, config.ndim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.ndim, config.ndim)
        self.out_ln = nn.LayerNorm(config.ndim)
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.out_ln(x)
        return x
    
