from config import VexConfig
import torch.multiprocessing as mp
from trainer.trainer import Trainer
# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     config = VexConfig()
#     config.train_device = "cuda:0"
#     config.inference_server_device = "cuda:0"
#     config.n_workers = 60

#     trainer = Trainer(config)
#     trainer.train()

import time 
config = VexConfig()
from model.model import AgentMLP
import torch 
agent = AgentMLP(config)
batch_size = 1
# benchmark batch of 64 for 1000 iterations. on cpu 
core_obs = torch.randn((batch_size, 2, config.core_obs_dim)).view(-1, config.core_obs_dim)
ball_obs = torch.randn((batch_size, 2, config.n_balls, config.ball_obs_dim)).view(-1, config.n_balls, config.ball_obs_dim)
legal_masks = torch.ones((batch_size, 2, config.n_primary_actions), dtype=torch.bool).view(-1, config.n_primary_actions)
agent.eval()
with torch.no_grad():
    start_time = time.time()
    for _ in range(1200):
        out = agent(core_obs, ball_obs, legal_masks, inference=True)
    end_time = time.time()
    print(f"Inference time for batch of 1: {end_time - start_time:.2f}s")