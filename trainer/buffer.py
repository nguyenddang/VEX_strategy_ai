import torch 

class SharedRolloutBuffer:
    
    def __init__(self, n_workers, obs_dim, action_dim, max_episode_length):
        self.obs = torch.zeros((2, n_workers, max_episode_length, obs_dim), dtype=torch.float32).share_memory_()
        self.actions = torch.zeros((2, n_workers, max_episode_length, action_dim), dtype=torch.float32).share_memory_()
        self.rewards = torch.zeros((2, n_workers, max_episode_length), dtype=torch.float32).share_memory_()
        
