from model.model import GeniusFormer
from config import VexConfig
import torch
import time

config = VexConfig()
model = GeniusFormer(config)

B = 2
total_timesteps = 100

start_time = time.time()

for _ in range(10):
    batch = {
        'core_obs': torch.randint(0, 10, (B*2, total_timesteps, config.core_obs_dim)), # 128, 600, 85
        'ball_obs': torch.randn(B*2, total_timesteps, config.n_balls, config.ball_obs_dim),
        'legal_masks': torch.randint(0, 2, (B*2*total_timesteps, config.n_primary_actions), dtype=torch.bool),
        'move_masks': torch.randint(0, 2, (B*2*total_timesteps, ), dtype=torch.bool),
    }
    core_obs = batch['core_obs']
    ball_obs = batch['ball_obs']
    legal_masks = batch['legal_masks']
    move_masks = batch['move_masks']

    padded_core_obs = torch.ones((B*2, config.block_size - 1, config.core_obs_dim)) * -1 
    core_obs = torch.cat([padded_core_obs, core_obs], dim=1) # (B*2, block_size - 1 + total_timesteps, core_obs_dim)
    batched_core_obs = core_obs.unfold(1, config.block_size, 1).transpose(2, 3)
    batched_core_obs = batched_core_obs.contiguous().view(-1, config.block_size, config.core_obs_dim) # (B*2*(total_timesteps - block_size + 1), block_size, core_obs_dim)

    padded_ball_obs = torch.ones((B*2, config.block_size - 1, config.n_balls, config.ball_obs_dim)) * -1
    ball_obs = torch.cat([padded_ball_obs, ball_obs], dim=1) 
    batched_ball_obs = ball_obs.unfold(1, config.block_size, 1).transpose(3, 4).transpose(2, 3)
    batched_ball_obs = batched_ball_obs.contiguous().view(-1, config.block_size, config.n_balls, config.ball_obs_dim)

    pads = torch.zeros((total_timesteps + config.block_size - 1, ), dtype=torch.bool)
    pads[:config.block_size-1] = 1
    padding_masks = pads.unfold(0, config.block_size, 1)

    outputs = model(batched_core_obs, batched_ball_obs, padding_masks)
    print(f"outputs shape: {outputs.shape}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for 10 iterations: {elapsed_time:.2f} seconds")
