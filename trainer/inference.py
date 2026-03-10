import torch 
import torch.nn.functional as F
from torch.func import functional_call

from trainer.shared import SharedBuffer, SharedLeague
from model.model import AgentMLP
from config import VexConfig

import time 
import math 
from collections import OrderedDict

class InferenceServer:
    
    def __init__(
        self, 
        buffer: SharedBuffer,
        league: SharedLeague,
        config: VexConfig,
    ):
        self.buffer = buffer
        self.league = league
        self.config = config
        self.device_id = self.config.inference_server_device

        self.model = AgentMLP(self.config)
        self.model.to(self.device_id)
        self.model.eval()
        
        n_model = math.ceil(self.config.n_workers * self.config.latest_ratio + 1) + 5 # addition 10 gives a bit of room in case workers switch back and forth. 
        self.inference_bank = torch.zeros((n_model, sum(p.numel() for p in self.model.parameters())), dtype=torch.float32, device=self.device_id)
        self.lru_cache = OrderedDict() # (f"{opp_idx, param_version}" -> row idx in inference_bank)
        self.n_model = n_model
        self.learner_param = torch.zeros((sum(p.numel() for p in self.model.parameters()),), dtype=torch.float32, device=self.device_id)
        self.learner_weights = None
        self.param_names = [name for name, _ in self.model.named_parameters()]
        self.param_shapes = [param.shape for param in self.model.parameters()]
        self.param_numels = [param.numel() for param in self.model.parameters()]
        
    def run(self):
        torch.set_num_threads(1)
        print(f"Inference server started on device {self.device_id}.")
        
        while True:
            active_workers = []
            start_time = time.perf_counter()
            while len(active_workers) < self.config.inference_batch_size:
                try:
                    time_left = self.config.inference_timeout - (time.perf_counter() - start_time)
                    w_req= self.buffer.inference_queue.get(timeout=max(0, time_left))
                    active_workers.append(w_req)
                except:
                    break
            if len(active_workers) == 0:
                continue
        
            self.forward_batch(active_workers)
            torch.cuda.synchronize(self.device_id) # make sure all gpu operations are done before signaling workers.
            for w_req in active_workers:
                self.buffer.res_events[w_req[0]].set()
            
    def gather_prep(self, active_workers):
        # batch up data according to opponent idx and param version
        wids, slots = [], []
        for w_id, opp_idx, param_version in active_workers:
            wids.append(w_id)
            key = (opp_idx, param_version)
            if key not in self.lru_cache:
                # load model parameters into inference bank and update lru cache. 
                if len(self.lru_cache) >= self.n_model:
                    _, slot_idx = self.lru_cache.popitem(last=False) # pop the least recently used item.
                else:
                    slot_idx = len(self.lru_cache)
                with self.league.lock:
                    param_data = self.league.param_bank[opp_idx]
                self.inference_bank[slot_idx].copy_(param_data, non_blocking=True)
                self.lru_cache[key] = slot_idx
            else: 
                # move the accessed item to the end
                self.lru_cache.move_to_end(key)
            slots.append(self.lru_cache[key])
            
        # gather batch data from buffer.temp
        core_obs_batch = self.buffer.temp['core_obs'][wids].to(self.device_id, non_blocking=True)
        ball_obs_batch = self.buffer.temp['ball_obs'][wids].to(self.device_id, non_blocking=True)
        legal_masks_batch = self.buffer.temp['legal_masks'][wids].to(self.device_id, non_blocking=True)
        unique_slots = list(set(slots))
        wids = torch.tensor(wids, dtype=torch.long)
        slots = torch.tensor(slots, dtype=torch.long)
        return wids, slots, unique_slots, core_obs_batch, ball_obs_batch, legal_masks_batch
    
    def reshape_params(self, flat_param):
        params = {}
        pointer = 0
        # load params. 
        for name, shape, numel in zip(self.param_names, self.param_shapes, self.param_numels):
            params[name] = flat_param[pointer:pointer+numel].view(shape)
            pointer += numel            
        return params
    
    def forward_batch(self, active_workers):
        wids, slots, unique_slots, core_obs_batch, ball_obs_batch, legal_masks_batch = self.gather_prep(active_workers)
        # check if learner has been updated
        if self.league.learner_updated_event.is_set():
            with self.league.learner_lock:
                self.learner_param.copy_(self.league.learner_param, non_blocking=True)
            self.league.learner_updated_event.clear()
            self.learner_weights = self.reshape_params(self.learner_param)
        # inference red first (idx=0). Same learner is used for all workers. 
        red_out = functional_call(
            self.model,
            self.learner_weights,
            (core_obs_batch[:, 0], ball_obs_batch[:, 0], legal_masks_batch[:, 0], True)
        )
        # save to temp
        self.buffer.temp['actions'][wids, 0].copy_(red_out['action'], non_blocking=True)
        self.buffer.temp['log_probs'][wids, 0].copy_(red_out['log_prob'], non_blocking=True)
        self.buffer.temp['values'][wids, 0].copy_(red_out['value'], non_blocking=True)
        self.buffer.temp['move_masks'][wids, 0].copy_(red_out['move_mask'], non_blocking=True)
        for slot_idx in unique_slots:
            slot_mask = (slots == slot_idx)
            if slot_mask.sum() == 0:
                continue
            opp_weights = self.reshape_params(self.inference_bank[slot_idx])
            with torch.no_grad():
                blue_out = functional_call(
                    self.model, 
                    opp_weights, 
                    (core_obs_batch[slot_mask][:, 1], ball_obs_batch[slot_mask][:, 1], legal_masks_batch[slot_mask][:, 1], True))
            self.buffer.temp['actions'][wids[slot_mask]][:, 1].copy_(blue_out['action'], non_blocking=True)
            self.buffer.temp['log_probs'][wids[slot_mask]][:, 1].copy_(blue_out['log_prob'], non_blocking=True)
            self.buffer.temp['values'][wids[slot_mask]][:, 1].copy_(blue_out['value'], non_blocking=True)
            self.buffer.temp['move_masks'][wids[slot_mask]][:, 1].copy_(blue_out['move_mask'], non_blocking=True)
            
    
        
        
        
                             
        
        
                    
            