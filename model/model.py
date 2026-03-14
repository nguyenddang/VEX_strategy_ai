from config import VexConfig
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from model.utils import Encoder
from model.transformer import Transformer

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


class GeniusFormer(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.encoder = Encoder(config)
        self.transformer = Transformer(config)

        self.register_buffer("attn_mask", self.create_attention_mask())

        self.apply(self._init_weights)
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M", flush=True)

    def create_attention_mask(self):
        causal_mask = torch.tril(torch.ones((self.config.block_size, self.config.block_size), dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        pads = torch.zeros((self.config.total_timesteps + self.config.block_size - 1, ), dtype=torch.bool)
        pads[:self.config.block_size] = 1
        padding_mask = pads.unfold(0, self.config.block_size, 1)

        attn_mask = causal_mask | padding_mask.unsqueeze(-1).expand(-1, -1, self.config.block_size).unsqueeze(0) 
        attn_mask = attn_mask.expand(self.config.mini_train_episodes*2, -1, -1, -1).contiguous().view(-1, 1, self.config.block_size, self.config.block_size)

        return attn_mask
    
    def reset_kv_cache(self):
        for block in self.transformer.transformers.h:
            block.attn.reset_cv_cache()

    def forward(self, core_obs, ball_obs, legal_masks, do_inference=False):
        """
        core_obs (variables): (B, T, core_obs_dim)
        ball_obs (ball entities): (B, T, n_balls, ball_obs_dim)
        legal_masks: (B, T, n_primary_actions)
        """

        if do_inference:
            if self.attn_mask is not None:
                delattr(self, "attn_mask")
                self.attn_mask = None

        x = self.encoder(core_obs, ball_obs) # (B, T, n_embd)
        assert x.shape[-1] == self.n_embd, f"Input embedding dimension {x.shape[-1]} does not match model dimension {self.n_embd}"

        policy_logits, value_logits = self.transformer(x, attn_mask=self.attn_mask if not do_inference else None)

        unmasked_primary_action_logits, x_bin_logits, y_bin_logits, theta_bin_logits = torch.split(policy_logits, [self.config.n_primary_actions, self.config.N, self.config.N, self.config.K], dim=-1)
        primary_action_logits = unmasked_primary_action_logits.masked_fill(legal_masks == 0, float("-inf"))

        out = {
            "primary_action_logits": primary_action_logits, # (B, n_primary_actions)
            "x_bin_logits": x_bin_logits, # (B, N)
            "y_bin_logits": y_bin_logits, # (B, N)
            "theta_bin_logits": theta_bin_logits, # (B, K)
            "value_logits": value_logits, # (B, 1)
        }

        return self.inference(out) if do_inference else out
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.088)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.088)
    
    def inference(self, out):
        """
        OUTPUT
        actions: (B, 4) - p, x, y, theta
        log_prob: (B, )
        values: (B, )
        move_mask: (B, ) - whether the action is move or not, used for value function learning. Only when primary action is MOVE, the x/y/theta bins matter.
        """

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
        theta_prob = theta_dist.log_prob(theta_act)

        move_mask = (p_act == 1) # only when primary action is MOVE, the x/y/theta bins matter.
        log_prob = p_prob + move_mask * (x_prob + y_prob + theta_prob) # (B)

        return {
            'actions': act,
            'log_prob': log_prob,
            'values': out["value_logits"].squeeze(-1), # (B)
            'move_mask': move_mask,
        }
        
    def reset_kv_cache(self):
        # set all self.k_cache, self.v_cache to in Attention module to None
        for block in self.transformer.transformers.h:
            block.attn.reset_kv_cache()
