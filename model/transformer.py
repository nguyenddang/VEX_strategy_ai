from config import VexConfig
import torch 
import torch.nn as nn


class CausalSelfAttention(nn.Module):

    def __init__(self, config: VexConfig):
        super().__init__()
        assert config.ndim % config.nhead == 0
        self.c_attn = nn.Linear(config.ndim, 3 * config.ndim, bias=config.bias)
        self.c_proj = nn.Linear(config.ndim, config.ndim, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.nhead = config.nhead
        self.ndim = config.ndim
        self.dropout = config.dropout
    def forward(self, x, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v  = self.c_attn(x).split(self.ndim, dim=2)
        k = k.view(B, T, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=True if attn_mask is None else False)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class MLP(nn.Module):

    def __init__(self, config: VexConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.ndim, 4 * config.ndim, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.ndim, config.ndim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = torch.nn.functional.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    
    def __init__(self, config: VexConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.ndim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.ndim)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x