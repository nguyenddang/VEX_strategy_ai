from trainer.shared import SharedBuffer
from trainer.inference import inference_server_fn
from trainer.worker import worker_fn
from config import VexConfig
import torch.multiprocessing as mp
import time 
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    config = VexConfig()
    buffer = SharedBuffer(config)
    # inference server process
    server_proc = mp.Process(target=inference_server_fn, args=(buffer, config, 'cuda'))
    server_proc.start()
    
    workers_proc = []
    for worker_id in range(config.n_workers):
        p = mp.Process(target=worker_fn, args=(worker_id, buffer, config))
        p.start()
        workers_proc.append(p)
        
    update_interval = 2.0  # We want to update the model every 2 seconds
    start_time = time.perf_counter()

    while True:
        # 1. Wait for the update interval (Simulating training time)
        time.sleep(update_interval)
        
        # 2. Check the "Pile"
        chunks_waiting = buffer.ready_indices.qsize()
        steps_waiting = chunks_waiting * config.chunk_size
        
        # 3. PULL EVERYTHING
        # This matches the worker production exactly
        actual_batch = buffer.pull_from_buffer(steps_waiting)
        
        # 4. Reporting
        elapsed = time.perf_counter() - start_time
        print(f"--- Interval: {update_interval}s ---")
        print(f"Required Batch Size to keep up: {steps_waiting}")
        print(f"Current System Production FPS: {steps_waiting / update_interval:.2f}")
        
        start_time = time.perf_counter()
