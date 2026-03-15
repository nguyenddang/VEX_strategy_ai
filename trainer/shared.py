import torch 
import torch.multiprocessing as mp

from config import VexConfig
from typing import Dict

from model.model import GeniusFormer
import time
import random 

class SharedBuffer:
    def __init__(self, config: VexConfig):
        self.buffer_capacity = config.buffer_capacity
        self.max_actions = config.max_actions
        self.mini_train_episodes = config.mini_train_episodes
        # 2 for red/blue robot. 
        # values: max_actions + 1 for bootstrapping value of last state.
        self.buffer = {
            'core_obs': torch.zeros((self.buffer_capacity, 2, self.max_actions, config.core_obs_dim), dtype=torch.float32).share_memory_(),
            'ball_obs': torch.zeros((self.buffer_capacity, 2, self.max_actions, config.n_balls, config.ball_obs_dim), dtype=torch.float32).share_memory_(),
            'legal_masks': torch.zeros((self.buffer_capacity, 2, self.max_actions, config.n_primary_actions), dtype=torch.bool).share_memory_(),
            'actions': torch.zeros((self.buffer_capacity, 2, self.max_actions, 4), dtype=torch.long).share_memory_(),
            'rewards': torch.zeros((self.buffer_capacity, 2, self.max_actions), dtype=torch.float32).share_memory_(),
            'values': torch.zeros((self.buffer_capacity, 2, self.max_actions + 1), dtype=torch.float32).share_memory_(),
            'move_masks': torch.zeros((self.buffer_capacity, 2, self.max_actions), dtype=torch.bool).share_memory_(),
            'log_probs': torch.zeros((self.buffer_capacity, 2, self.max_actions), dtype=torch.float32).share_memory_(),
            'learner_versions': torch.zeros((self.buffer_capacity, self.max_actions), dtype=torch.float32).share_memory_(), # TODO: change this to take only a single version instead of the the total episodes
            'red_score': torch.zeros((self.buffer_capacity, 1), dtype=torch.float32).share_memory_(),
        }
        
        self.read_write_queue = mp.Queue() # worker pull from queue to write, trainer gpu pull from queue to read. ensures no overlapping write/read.
        self.written_before = torch.zeros((self.buffer_capacity,), dtype=torch.bool).share_memory_() # to track which slots have been written to before for sampling.
        # Initialize queue with all buffer indices
        for i in range(self.buffer_capacity):
            self.read_write_queue.put(i)
            
        # track producer-consumer speed.
        self.sample_produced = mp.Value('l', 0)
        
    @torch.no_grad()
    def push_to_buffer(self, data_chunk: Dict[str, torch.Tensor]):
        # called by worker to push a chunk of data to the buffer.
        # ensure workers cannot write to the same slot at the same time. 
        idx = self.read_write_queue.get() # block until get a free slot index.
        for key in self.buffer:
            self.buffer[key][idx].copy_(data_chunk[key])
        self.written_before[idx] = True
        self.read_write_queue.put(idx) 
        with self.sample_produced.get_lock():
            self.sample_produced.value += self.max_actions * 2
        
    def pull_from_buffer(self):
        # called by trainer gpu to pull a batch of data from the buffer.
        ids = []
        batch = {k: None for k in self.buffer}
        while torch.sum(self.written_before) < self.buffer_capacity:
            time.sleep(0.05) # wait until enough episodes are written by workers.
        while len(ids) < self.mini_train_episodes:
            idx = self.read_write_queue.get() # block until enough chunks are ready.
            ids.append(idx)
        for key in self.buffer:
            batch[key] = self.buffer[key][ids]
        for idx in ids:
            self.read_write_queue.put(idx)
        return batch
    

class SharedLeague:
    def __init__(self, config: VexConfig):
        self.max_league_snapshots = config.max_league_snapshots
        self.latest_ratio = config.latest_ratio
        self.n_workers = config.n_workers
        
        local_model = GeniusFormer(config)
        total_params = sum(p.numel() for p in local_model.parameters())
        del local_model
        self.latest_opp_idx = mp.Value('i', -1) # index of the latest snapshot in the bank.
        self.opp_bank = torch.zeros((self.max_league_snapshots, total_params), dtype=torch.float32).share_memory_() #league snapshots.
        self.opp_qualities = torch.full((self.max_league_snapshots,), float('-inf'), dtype=torch.float32).share_memory_() #quality of league snapshots. higher better.
        self.opp_just_updated = torch.zeros((self.max_league_snapshots,), dtype=torch.bool).share_memory_() 
        self.opp_lock = mp.Lock()
        
        self.learner_param = torch.zeros((total_params,), dtype=torch.float32).share_memory_()
        self.learner_version = mp.Value('i', -1)
        self.learner_lock = mp.Lock()
        
    def update_latest_snapshot(self, model: GeniusFormer):
        # push newest snapshot into the bank. 
        # called by trainer gpu after each n (every config.steps_per_iteration) 
        param = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu()
        with self.opp_lock:
            if self.latest_opp_idx.value == -1:
                # first snapshot, directly put in first splot
                replace_idx = 0
                new_quality = 0
            else:
                replace_idx = torch.argmin(self.opp_qualities).item()
                new_quality = torch.max(self.opp_qualities).item() # set new quality to highest in quality in the bank. 
            self.opp_qualities[replace_idx] = new_quality
            self.latest_opp_idx.value = replace_idx
            self.opp_bank[replace_idx].copy_(param)
            self.opp_just_updated[replace_idx] = True
            
    def update_learner_param(self, model: GeniusFormer):
        param = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu()
        with self.learner_lock:
            self.learner_param.copy_(param)
            self.learner_version.value += 1
        
    def sample_opponent(self, worker_id: int):
        with self.opp_lock:
            valid_count = (self.opp_qualities > float('-inf')).sum().item()
            assert valid_count >= 1, "[LEAGUE] There should be at least one valid snapshot in the league to sample."
            # 80% case or if the bank only has one model
            if worker_id < int(self.n_workers * self.latest_ratio) or valid_count <= 1:
                idx = self.latest_opp_idx.value
                self.opp_just_updated[idx] = False
                return idx, 1.0, valid_count, self.opp_bank[idx]
            # 20% case: sample from the rest
            q = self.opp_qualities.clone()
            q[self.latest_opp_idx.value] = float('-inf')
            probs = torch.softmax(q, dim=0)
            opponent_idx = torch.multinomial(probs, num_samples=1).item()
            p = probs[opponent_idx]
            self.opp_just_updated[opponent_idx] = False
        return opponent_idx, p, valid_count, self.opp_bank[opponent_idx]
    
    def update_quality(self, opponent_idx: int, delta:float):
        # called by worker to update quality of a snapshot after selfplay. 
        with self.opp_lock:
            if not self.opp_just_updated[opponent_idx]:
                self.opp_qualities[opponent_idx] -= delta

    def state_dict(self):
        return {
            "latest_opp_idx": self.latest_opp_idx.value,
            "opp_bank": self.opp_bank.clone(),
            "opp_qualities": self.opp_qualities.clone(),
            "opp_just_updated": self.opp_just_updated.clone(),
            "learner_param": self.learner_param.clone(),
            "learner_version": self.learner_version.value
        }
    
    def load_state_dict(self, state_dict):
        with self.opp_lock:
            self.latest_opp_idx.value = state_dict["latest_opp_idx"]
            self.opp_bank.copy_(state_dict["opp_bank"])
            self.opp_qualities.copy_(state_dict["opp_qualities"])
            self.opp_just_updated.copy_(state_dict["opp_just_updated"])
        with self.learner_lock:
            self.learner_param.copy_(state_dict["learner_param"])
            self.learner_version.value = state_dict["learner_version"]
            