import torch 
import torch.nn as nn
from torch.distributions import Categorical

from model.utils import Encoder

from config import VexConfig
class MLP(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        self.encoder  = Encoder(config)
        self.base = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd*2),
            nn.GELU(),
            nn.LayerNorm(config.n_embd*2),
            nn.Linear(config.n_embd*2, config.n_embd*2),
            nn.GELU(),
            nn.LayerNorm(config.n_embd*2),
        )
        self.head = nn.Linear(config.n_embd*2, config.n_primary_actions + config.N * 2 + config.K + 1)
        self.config = config
        print(f"MLP initialized with {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters.")
        
    def forward(self, core_obs, ball_obs, legal_masks, inference=False):
        # core_obs: (B, core_obs_dim)
        # ball_obs: (B, n_balls, ball_obs_dim)
        # legal_masks: (B, n_primary_actions) 
        x = self.encoder(core_obs, ball_obs) # (B, n_embd)
        x = self.base(x)
        policy_value_logits = self.head(x)

        unmasked_primary_action_logits, x_bin_logits, y_bin_logits, theta_bin_logits, value_logits = \
            torch.split(
                policy_value_logits, 
                [self.config.n_primary_actions, 
                 self.config.N, 
                 self.config.N, 
                 self.config.K, 
                 1], dim=-1)
        primary_action_logits = unmasked_primary_action_logits.masked_fill(legal_masks == 0, float("-inf"))
        out = {
            "primary_action_logits": primary_action_logits, # (B, n_primary_actions)
            "x_bin_logits": x_bin_logits, # (B, N)
            "y_bin_logits": y_bin_logits, # (B, N)
            "theta_bin_logits": theta_bin_logits, # (B, K)
            "value_logits": value_logits.squeeze(-1), # (B)
        }

        return self.inference(out) if inference else out
    
    def inference(self, out):
        p_dist = Categorical(logits=out["primary_action_logits"])
        x_dist = Categorical(logits=out["x_bin_logits"])
        y_dist = Categorical(logits=out["y_bin_logits"])
        theta_dist = Categorical(logits=out["theta_bin_logits"])
        
        p_act = p_dist.sample() # (B)
        x_act = x_dist.sample() # (B)
        y_act = y_dist.sample() # (B)
        theta_act = theta_dist.sample() # (B)
        
        act = torch.stack([p_act, x_act, y_act, theta_act], dim=-1) # (B, 4)
        p_prob = p_dist.log_prob(p_act) # (B)
        x_prob = x_dist.log_prob(x_act) # (B)
        y_prob = y_dist.log_prob(y_act) # (B)
        theta_prob = theta_dist.log_prob(theta_act) # (B)
        
        move_mask = (p_act == 0) # only when primary action is MOVE, the x/y/theta bins matter.
        log_prob = p_prob + move_mask * (x_prob + y_prob + theta_prob) # (B)
        return {
            'actions': act,
            'log_prob': log_prob,
            'values': out["value_logits"], # (B)
            'move_mask': move_mask, # (B)
        }