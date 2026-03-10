from config import VexConfig
from model.transformer import Block
from model.utils import BallEncoder, CoreEncoder, BasicEncoder

import torch 
import torch.nn as nn
from torch.distributions import Categorical
        
class AgentMLP(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        self.config = config 
        self.core_encoder = CoreEncoder(config)
        self.ball_encoder = BallEncoder(config)
        self.basic_encoder = BasicEncoder(config)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.ndim, config.ndim * 4),
            nn.ReLU(),
            nn.Linear(config.ndim * 4, config.ndim * 4),
            nn.ReLU(),
            nn.Linear(config.ndim * 4, config.ndim * 4),
            nn.ReLU(),
            nn.Linear(config.ndim * 4, config.ndim * 4),
            nn.ReLU(),
            nn.Linear(config.ndim * 4, config.n_primary_actions + config.N * 2 + config.K + 1)
        )
        
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
    
    def forward(self, core_obs, ball_obs, legal_action_mask=None, inference=False):
        # core_obs: (B, core_obs_dim).
        # ball_obs: (B, n_ball, ball_obs_dim)
        core_embed = self.core_encoder(core_obs) # (B, ndim//2)
        ball_embed = self.ball_encoder(ball_obs) # (B, ndim//2)
        x = torch.cat([core_embed, ball_embed], dim=-1) # (B, ndim)
        x = self.basic_encoder(x) # (B, ndim)
        
        x = self.mlp(x) # (B, n_primary_actions + N * 2 + K + 1)
        primary_action_logits, x_bin_logits, y_bin_logits, theta_bin_logits, value_logits = torch.split(x, [self.config.n_primary_actions, self.config.N, self.config.N, self.config.K, 1], dim=-1)
        if legal_action_mask is not None:
            primary_action_logits = primary_action_logits.masked_fill(legal_action_mask == 0, float("-inf"))
            
        out = {
            "primary_action_logits": primary_action_logits, # (B, n_primary_actions)
            "x_bin_logits": x_bin_logits, # (B, N)
            "y_bin_logits": y_bin_logits, # (B, N)
            "theta_bin_logits": theta_bin_logits, # (B, K)
            "value_logits": value_logits, # (B, 1)
        }
        if self.inference:
            return self.inference(out)
        return out
        
    def inference(self, outputs):
        p_dist = Categorical(logits=outputs["primary_action_logits"])
        x_dist = Categorical(logits=outputs["x_bin_logits"])
        y_dist = Categorical(logits=outputs["y_bin_logits"])
        theta_dist = Categorical(logits=outputs["theta_bin_logits"])
        
        p_act = p_dist.sample() # (B)
        x_act = x_dist.sample() # (B)
        y_act = y_dist.sample() # (B)
        theta_act = theta_dist.sample() # (B)
        act = torch.stack([p_act, x_act, y_act, theta_act], dim=-1) # (B, 4)
        
        p_prob = p_dist.log_prob(p_act) # (B)
        x_prob = x_dist.log_prob(x_act) # (B)
        y_prob = y_dist.log_prob(y_act) # (B)
        theta_prob = theta_dist.log_prob(theta_act)
        move_mask = (p_act == 1) # only when primary action is MOVE, the x/y/theta bins matter.
        log_prob = p_prob + move_mask * (x_prob + y_prob + theta_prob) # (B)
        return {
            'action': act,
            'log_prob': log_prob,
            'value': outputs["value_logits"].squeeze(-1), # (B)
            'move_mask': move_mask,
        }
        

        
        
        
        