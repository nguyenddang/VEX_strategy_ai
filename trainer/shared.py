import torch 
import torch.multiprocessing as mp

from config import VexConfig
from typing import Dict

from model.model import AgentMLP
import time

class SharedBuffer:
    def __init__(self, config: VexConfig):
        self.buffer_capacity = config.buffer_capacity
        self.chunk_size = config.chunk_size
        self.train_batch_size = config.train_batch_size
        # 2 for red/blue robot. 
        # values: chunk_size + 1 for bootstrapping value of last state.
        self.buffer = {
            'core_obs': torch.zeros((self.buffer_capacity, 2, self.chunk_size, config.core_obs_dim), dtype=torch.float32).share_memory_(),
            'ball_obs': torch.zeros((self.buffer_capacity, 2, self.chunk_size, config.n_balls, config.ball_obs_dim), dtype=torch.float32).share_memory_(),
            'legal_masks': torch.zeros((self.buffer_capacity, 2, self.chunk_size, config.n_primary_actions), dtype=torch.bool).share_memory_(),
            'actions': torch.zeros((self.buffer_capacity, 2, self.chunk_size, 4), dtype=torch.long).share_memory_(),
            'rewards': torch.zeros((self.buffer_capacity, 2, self.chunk_size), dtype=torch.float32).share_memory_(),
            'values': torch.zeros((self.buffer_capacity, 2, self.chunk_size + 1), dtype=torch.float32).share_memory_(),
            'move_masks': torch.zeros((self.buffer_capacity, 2, self.chunk_size), dtype=torch.bool).share_memory_(),
            'log_probs': torch.zeros((self.buffer_capacity, 2, self.chunk_size), dtype=torch.float32).share_memory_(),
        }
        
        # Buffer also handles comm between worker and inference gpu. 
        self.temp = {
            'core_obs': torch.zeros((config.n_workers, 2, config.core_obs_dim), dtype=torch.float32).share_memory_(),
            'ball_obs': torch.zeros((config.n_workers, 2, config.n_balls, config.ball_obs_dim), dtype=torch.float32).share_memory_(),
            'legal_masks': torch.zeros((config.n_workers, 2, config.n_primary_actions), dtype=torch.float32).share_memory_(),
            'actions': torch.zeros((config.n_workers, 2, 4), dtype=torch.long).share_memory_(),
            'log_probs': torch.zeros((config.n_workers, 2), dtype=torch.float32).share_memory_(),
            'values': torch.zeros((config.n_workers, 2), dtype=torch.float32).share_memory_(),
            'move_masks': torch.zeros((config.n_workers, 2), dtype=torch.bool).share_memory_(),
        } 
        self.inference_queue = mp.Queue() # worker gpu puts inference requests (worker ids).
        self.res_events = [mp.Event() for _ in range(config.n_workers)] # set by inference gpu when results are ready.
        self.read_write_queue = mp.Queue() # worker pull from queue to write, trainer gpu pull from queue to read. ensures no overlapping write/read.
        
        self.written_before = torch.zeros((self.buffer_capacity,), dtype=torch.bool).share_memory_() # to track which slots have been written to before for sampling.
        # Initialize queue with all buffer indices
        for i in range(self.buffer_capacity):
            self.read_write_queue.put(i)
            
        # track producer-consumer speed.
        self.sample_produced = mp.Value('l', 0)
        self.sample_read_lock = mp.Lock()
    def push_to_buffer(self, data_chunk: Dict[str, torch.Tensor]):
        # called by worker to push a chunk of data to the buffer.
        # ensure workers cannot write to the same slot at the same time. 
        idx = self.read_write_queue.get() # block until get a free slot index.
        for key in self.buffer:
            self.buffer[key][idx].copy_(data_chunk[key])
        self.written_before[idx] = True
        self.read_write_queue.put(idx) 
        with self.sample_read_lock:
            self.sample_produced.value += self.chunk_size * 2
        
    def pull_from_buffer(self):
        # called by trainer gpu to pull a batch of data from the buffer.
        batch = {k: [] for k in self.buffer}
        n_chunks = self.train_batch_size // self.chunk_size
        while torch.sum(self.written_before) < n_chunks:
            time.sleep(0.01) # wait until enough chunks are written by workers.
        while len(batch['core_obs']) < n_chunks:
            idx = self.read_write_queue.get() # block until enough chunks are ready.
            if not self.written_before[idx]:
                self.read_write_queue.put(idx) # put back if never been written to and sleep a bit
                continue
            for key in self.buffer:
                batch[key].append(self.buffer[key][idx]) 
            self.read_write_queue.put(idx) # give back to queue 
        for key in batch:
            batch[key] = torch.stack(batch[key], dim=0)
        return batch
    

class SharedLeague:
    def __init__(self, config: VexConfig):
        self.max_league_snapshots = config.max_league_snapshots
        self.latest_ratio = config.latest_ratio
        self.n_workers = config.n_workers
        
        local_model = AgentMLP(config)
        total_params = sum(p.numel() for p in local_model.parameters())
        del local_model
        self.latest_idx = mp.Value('i', 0) # index of the latest snapshot in the bank.
        self.global_counter = mp.Value('i', 0) # used to get unique id for each snapshot version.
        self.param_bank = torch.zeros((self.max_league_snapshots, total_params), dtype=torch.float32).share_memory_() # store model parameters of league snapshots.
        self.param_version = torch.zeros((self.max_league_snapshots,), dtype=torch.long).share_memory_() 
        self.qualities = torch.full((self.max_league_snapshots,), float('-inf'), dtype=torch.float32).share_memory_() # store quality of league snapshots. higher is better.
        self.lock = mp.Lock()
        
        self.learner_param = torch.zeros((total_params,), dtype=torch.float32).share_memory_()
        self.learner_updated_event = mp.Event() # set by trainer gpu when new learner has been added
        self.learner_lock = mp.Lock()
        
    def update_latest_snapshot(self, model: AgentMLP):
        # push newest snapshot into the bank. 
        # called by trainer gpu after each optimizer step. 
        param = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu()
        with self.lock:
            if self.global_counter.value == 0:
                # first snapshot, directly put in first splot
                replace_idx = 0
                new_quality = 0
            else:
                replace_idx = torch.argmin(self.qualities).item()
                new_quality = torch.max(self.qualities).item() # set new quality to highest in quality in the bank. 
            self.global_counter.value += 1
            self.qualities[replace_idx] = new_quality
            self.latest_idx.value = replace_idx
            self.param_bank[replace_idx].copy_(param)
            self.param_version[replace_idx] = self.global_counter.value
            
    def update_learner_param(self, model: AgentMLP):
        param = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu()
        with self.learner_lock:
            self.learner_param.copy_(param)
        self.learner_updated_event.set() # signal that learner param has been updated.
            
        
    def sample_opponent(self, worker_id: int):
        with self.lock:
            valid_count = (self.qualities > float('-inf')).sum().item()
            # 80% case or if the bank only has one model
            if worker_id < int(self.n_workers * self.latest_ratio) or valid_count <= 1:
                idx = self.latest_idx.value
                return idx, self.param_version[idx].item(), 1.0, self.qualities[idx].item(), valid_count
            # 20% case: sample from the rest
            q = self.qualities.clone()
            q[self.latest_idx.value] = float('-inf')
            probs = torch.softmax(q, dim=0)
            opponent_idx = torch.multinomial(probs, num_samples=1).item()
            param_version = self.param_version[opponent_idx].item()
            p = probs[opponent_idx]
            old_q = self.qualities[opponent_idx]
            
        return opponent_idx, param_version, p, old_q, valid_count
    
    def update_quality(self, opponent_idx: int, param_version: int, new_q: torch.Tensor):
        # called by worker to update quality of a snapshot after selfplay. 
        with self.lock:
            if param_version != self.param_version[opponent_idx].item():
                # snapshot has been updated, skip. 
                return False
            self.qualities[opponent_idx].copy_(new_q)
        return True

        
        
            
        
            