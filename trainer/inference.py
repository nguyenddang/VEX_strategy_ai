import torch 
import torch.nn.functional as F

from trainer.shared import SharedBuffer
from model.model import AgentMLP
from config import VexConfig

import time 

def inference_server_fn(
    buffer: SharedBuffer,
    config: VexConfig,
    device_id: str,
):
    torch.set_num_threads(1)
    assert torch.cuda.is_available(), "CUDA is not available"
    model = AgentMLP(config).to(device_id)
    model.eval()
    print(f"Inference server started on device {device_id}.")
    
    while True:
        active_worker_ids = []
        start_time = time.perf_counter()
        
        while len(active_worker_ids)  < config.inference_batch_size:
            # wait til batch size reached or timeout. 
            try:
                time_left = config.inference_timeout - (time.perf_counter() - start_time)
                w_id = buffer.inference_queue.get(timeout=max(0, time_left))
                active_worker_ids.append(w_id)
            except:
                break
        if len(active_worker_ids) == 0:
            continue
        
        # gather batch data from buffer.slots
        core_obs_batch = buffer.slots['core_obs'][active_worker_ids].view(-1, config.core_obs_dim).to(device_id, non_blocking=True)
        ball_obs_batch = buffer.slots['ball_obs'][active_worker_ids].view(-1, config.n_balls, config.ball_obs_dim).to(device_id, non_blocking=True)
        legal_masks_batch = buffer.slots['legal_masks'][active_worker_ids].view(-1, config.n_primary_actions).to(device_id, non_blocking=True)
        
        out = model.inference(core_obs_batch, ball_obs_batch, legal_masks_batch)
        buffer.slots['actions'][active_worker_ids].copy_(out['action'].view(len(active_worker_ids), 2, -1).cpu())
        buffer.slots['log_probs'][active_worker_ids].copy_(out['log_prob'].view(len(active_worker_ids), 2).cpu())
        buffer.slots['values'][active_worker_ids].copy_(out['value'].view(len(active_worker_ids), 2).cpu())
        buffer.slots['move_masks'][active_worker_ids].copy_(out['move_mask'].view(len(active_worker_ids), 2).cpu())
        
        # release workers
        for w_id in active_worker_ids:
            buffer.res_events[w_id].set()