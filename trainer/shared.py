import torch 
import torch.multiprocessing as mp

from config import VexConfig
from typing import Dict

class SharedBuffer:
    
    def __init__(self, config: VexConfig):
        self.buffer_capacity = config.buffer_capacity
        self.chunk_size = config.chunk_size
        # 2 for red/blue robot. 
        # values: chunk_size + 1 for bootstrapping value of last state.
        self.buffer = {
            'core_obs': torch.zeros((self.buffer_capacity, 2, self.chunk_size, config.core_obs_dim), dtype=torch.float32).pin_memory().share_memory_(),
            'ball_obs': torch.zeros((self.buffer_capacity, 2, self.chunk_size, config.n_balls, config.ball_obs_dim), dtype=torch.float32).pin_memory().share_memory_(),
            'legal_masks': torch.zeros((self.buffer_capacity, 2, self.chunk_size, config.n_primary_actions), dtype=torch.bool).pin_memory().share_memory_(),
            'actions': torch.zeros((self.buffer_capacity, 2, self.chunk_size, 4), dtype=torch.long).pin_memory().share_memory_(),
            'rewards': torch.zeros((self.buffer_capacity, 2, self.chunk_size), dtype=torch.float32).pin_memory().share_memory_(),
            'values': torch.zeros((self.buffer_capacity, 2, self.chunk_size + 1), dtype=torch.float32).pin_memory().share_memory_(),
            'move_masks': torch.zeros((self.buffer_capacity, 2, self.chunk_size), dtype=torch.bool).pin_memory().share_memory_(),
        }
        
        self.free_indices = mp.Queue() # indices of free chunks in the buffer.
        for i in range(self.buffer_capacity):
            self.free_indices.put(i)
        self.ready_indices = mp.Queue() # indices of chunks ready for trainer gpu to consume.
        
        # Buffer also handles comm between worker and inference gpu. 
        self.slots = {
            'core_obs': torch.zeros((config.n_workers, 2, config.core_obs_dim), dtype=torch.float32).pin_memory().share_memory_(),
            'ball_obs': torch.zeros((config.n_workers, 2, config.n_balls, config.ball_obs_dim), dtype=torch.float32).pin_memory().share_memory_(),
            'legal_masks': torch.zeros((config.n_workers, 2, config.n_primary_actions), dtype=torch.float32).pin_memory().share_memory_(),
            'actions': torch.zeros((config.n_workers, 2, 4), dtype=torch.long).pin_memory().share_memory_(),
            'log_probs': torch.zeros((config.n_workers, 2), dtype=torch.float32).pin_memory().share_memory_(),
            'values': torch.zeros((config.n_workers, 2), dtype=torch.float32).pin_memory().share_memory_(),
            'move_masks': torch.zeros((config.n_workers, 2), dtype=torch.bool).pin_memory().share_memory_(),
        } 
        self.inference_queue = mp.Queue() # worker gpu puts inference requests (worker ids). 
        self.res_events = [mp.Event() for _ in range(config.n_workers)] # set by infrence gpu when results are ready. 
        
    def push_to_buffer(self, data_chunk: Dict[str, torch.Tensor]):
        # called by worker to push a chunk of data to the buffer.
        try:
            # get free index. 
            idx = self.free_indices.get_nowait()
            # copy data into buffer 
            for key, tensor in data_chunk.items():
                assert key in self.buffer, f"Invalid key {key} in data_chunk"
                self.buffer[key][idx].copy_(tensor)
            # mark index as ready for trainer gpu.
            self.ready_indices.put(idx)
        except:
            # no free slots, drop the chunk and move on. 
            pass 
        
    def pull_from_buffer(self, batch_size: int):
        # called by trainer gpu to pull a batch of data from the buffer.
        indicies = []
        n_chunks = batch_size // self.chunk_size
        for _ in range(n_chunks):
            indicies.append(self.ready_indices.get()) # block until enough chunks are ready.
        batch = {k: self.buffer[k][indicies] for k in self.buffer}
        for idx in indicies:
            self.free_indices.put(idx) # mark chunks as free after pulling.
        return batch