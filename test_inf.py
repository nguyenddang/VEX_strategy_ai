from model.model import AgentMLP
from config import VexConfig
import torch

config = VexConfig()
ms = [AgentMLP(config).to('cuda') for _ in range(20)]
core_obs_dummy = torch.randn((64 * 2 * 16, config.core_obs_dim), device='cuda')
ball_obs_dummy = torch.randn((64 * 2 * 16, config.n_balls, config.ball_obs_dim), device='cuda')
legal_mask_dummy = torch.ones((64 * 2 * 16, config.n_primary_actions), device='cuda')
with torch.no_grad():
    for m in ms:
        out = m(core_obs_dummy, ball_obs_dummy, legal_mask_dummy)
