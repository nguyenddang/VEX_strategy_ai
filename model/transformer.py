from config import VexConfig
import torch 
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, config: VexConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.n_embd, config.n_embd * 3)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.k_cache = None
        self.v_cache = None

    def kv_cache(self, k, v):
        if self.k_cache is None and self.v_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)[:, :, -self.config.block_size:, :]
            self.v_cache = torch.cat([self.v_cache, v], dim=2)[:, :, -self.config.block_size:, :]
        return self.k_cache, self.v_cache
    
    def reset_cv_cache(self):
        self.k_cache = None
        self.v_cache = None


    # x: B, T, n_embd
    # padding_mask: timesteps, T
    def forward(self, x, attn_mask):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, self.config.n_embd, dim=-1)
        assert C // self.config.n_head > 0, f"Embedding dimension {C} is not divisible by number of heads {self.config.n_head}"
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) # (B, n_head, T, head_dim)

        if not self.training:
            k, v = self.kv_cache(k, v)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=attn_mask is None, attn_mask=attn_mask)
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
    def forward(self, x, attn_mask): 
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
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

        self.policy_head = nn.Linear(config.n_embd, config.n_primary_actions + config.N*2 + config.K)
        self.value_head = nn.Linear(config.n_embd, 1)

    
    # B, T, n_embd
    def forward(self, x, attn_mask):
        B, T, C = x.size()
        positions = torch.arange(T, device=x.device) # (1, T)
        position_embd = self.transformers.wpe(positions) # (1, T, n_embd)
        x = x + position_embd # (B, T, n_embd)

        for block in self.transformers.h:
            x = block(x, attn_mask=attn_mask)
        
        x = self.transformers.ln_f(x) # (B, T, n_embd)
        x = x[:, -1, :]
        policy_logits = self.policy_head(x) # (B, T, action_size)
        value_logits = self.value_head(x) # (B, T, 1)
        return policy_logits, value_logits