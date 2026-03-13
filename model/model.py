from config import VexConfig
from model.transformer import Block
# from model.utils import BallEncoder, CoreEncoder, BasicEncoder

import torch 
import torch.nn as nn
from torch.distributions import Categorical

import torch.nn.functional as F
        
# class AgentMLP(nn.Module):
#     def __init__(self, config: VexConfig):
#         super().__init__()
#         self.config = config 
#         self.core_encoder = CoreEncoder(config)
#         self.ball_encoder = BallEncoder(config)
#         self.basic_encoder = BasicEncoder(config)
        
#         self.mlp = nn.Sequential(
#             nn.Linear(config.ndim, config.ndim * 4),
#             nn.ReLU(),
#             nn.Linear(config.ndim * 4, config.ndim * 4),
#             nn.ReLU(),
#             nn.Linear(config.ndim * 4, config.ndim * 4),
#             nn.ReLU(),
#             nn.Linear(config.ndim * 4, config.ndim * 4),
#             nn.ReLU(),
#             nn.Linear(config.ndim * 4, config.n_primary_actions + config.N * 2 + config.K + 1)
#         )
        
#         print(f"Total number of parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
    
#     def forward(self, core_obs, ball_obs, legal_action_mask=None, inference=False):
#         # core_obs: (B, core_obs_dim).
#         # ball_obs: (B, n_ball, ball_obs_dim)
#         core_embed = self.core_encoder(core_obs) # (B, ndim//2)
#         ball_embed = self.ball_encoder(ball_obs) # (B, ndim//2)
#         x = torch.cat([core_embed, ball_embed], dim=-1) # (B, ndim)
#         x = self.basic_encoder(x) # (B, ndim)
        
#         x = self.mlp(x) # (B, n_primary_actions + N * 2 + K + 1)
#         primary_action_logits, x_bin_logits, y_bin_logits, theta_bin_logits, value_logits = torch.split(x, [self.config.n_primary_actions, self.config.N, self.config.N, self.config.K, 1], dim=-1)
#         if legal_action_mask is not None:
#             primary_action_logits = primary_action_logits.masked_fill(legal_action_mask == 0, float("-inf"))
            
#         out = {
#             "primary_action_logits": primary_action_logits, # (B, n_primary_actions)
#             "x_bin_logits": x_bin_logits, # (B, N)
#             "y_bin_logits": y_bin_logits, # (B, N)
#             "theta_bin_logits": theta_bin_logits, # (B, K)
#             "value_logits": value_logits, # (B, 1)
#         }
#         if inference:
#             return self.inference(out)
#         return out
        
#     @torch.no_grad()
#     def inference(self, outputs):
#         p_dist = Categorical(logits=outputs["primary_action_logits"])
#         x_dist = Categorical(logits=outputs["x_bin_logits"])
#         y_dist = Categorical(logits=outputs["y_bin_logits"])
#         theta_dist = Categorical(logits=outputs["theta_bin_logits"])
        
#         p_act = p_dist.sample() # (B)
#         x_act = x_dist.sample() # (B)
#         y_act = y_dist.sample() # (B)
#         theta_act = theta_dist.sample() # (B)
#         act = torch.stack([p_act, x_act, y_act, theta_act], dim=-1) # (B, 4)
#         p_prob = p_dist.log_prob(p_act) # (B)
#         x_prob = x_dist.log_prob(x_act) # (B)
#         y_prob = y_dist.log_prob(y_act) # (B)
#         theta_prob = theta_dist.log_prob(theta_act)
#         move_mask = (p_act == 1) # only when primary action is MOVE, the x/y/theta bins matter.
#         log_prob = p_prob + move_mask * (x_prob + y_prob + theta_prob) # (B)
#         return {
#             'actions': act,
#             'log_prob': log_prob,
#             'values': outputs["value_logits"].squeeze(-1), # (B)
#             'move_mask': move_mask,
#         }
    

"""
Global:
- match time: 1
- score diff: 1
- controls: 4
Learner:
- position x, and y: 2
- orientation cos and sin: 2
- inventory: 1
- past action: 1 -> n_embd
- score: 1
- legal actions: 6
Opponent:
- position x, and y: 2
- orientation cos and sin: 2
- score: 1
Each ball:
- position x, and y: 2
- relative distance (dx, dy, dist): 3
- color: 1
Ball types: 88 -> + n_embd
"""

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
        x = self.gelu(self.fc_2(x)) # (B, block_size, n_embd//2)
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
        x = self.gelu(self.fc_2(x))
        x = self.ln(x)
        return x

class GeniusFormer(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.encoder = Encoder(config)
        self.transformer = Transformer(config)

        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def forward(self, core_obs, ball_obs, padding_masks):
        """
        core_obs (variables): (B, T, core_obs_dim)
        ball_obs (ball entities): (B, T, n_balls, ball_obs_dim)
        """
        x = self.encoder(core_obs, ball_obs) # (B, T, n_embd)
        assert x.shape[-1] == self.n_embd, f"Input embedding dimension {x.shape[-1]} does not match model dimension {self.n_embd}"
        logits = self.transformer(x, padding_mask=padding_masks) # (B, T, action_size)
        return logits


class Attention(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.n_embd, config.n_embd * 3)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("causal_mask", torch.tril(torch.ones((config.block_size, config.block_size), dtype=torch.bool)).unsqueeze(0).unsqueeze(0)) # (1, 1, block_size, block_size)

    # B, T, n_embd
    def forward(self, x, padding_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, self.config.n_embd, dim=-1)
        assert C // self.config.n_head > 0, f"Embedding dimension {C} is not divisible by number of heads {self.config.n_head}"
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        
        if padding_mask is not None:
            attn_mask = self.causal_mask | padding_mask.unsqueeze(-1).expand(-1, -1, T).unsqueeze(0) # (B, 1, T, T)
            attn_mask = attn_mask.expand(B // self.config.total_timestep, -1, -1, -1).contiguous().view(B, 1, T, T)
        else:
            attn_mask = self.causal_mask

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask) # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        
        return y
    
class Block(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.n_embd))
    
    # B, T, n_embd
    def forward(self, x, padding_mask=None): 
        x = x + self.attn(self.ln_1(x), padding_mask=padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        self.config = config

        self.transformers = nn.ModuleDict(dict(
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd), 
        ))

        self.lm_head = nn.Linear(config.n_embd, config.action_size)
    
    # B, T, n_embd
    def forward(self, x, padding_mask=None):
        B, T, C = x.size()
        positions = torch.arange(T, device=x.device).view(1, T) # (1, T)
        position_embd = self.transformers.wpe(positions) # (1, T, n_embd)
        x = x + position_embd # (B, T, n_embd)

        for block in self.transformers.h:
            x = block(x, padding_mask=padding_mask)
        
        x = self.transformers.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, action_size)
        return logits