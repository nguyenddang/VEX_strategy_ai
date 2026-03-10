from trainer.shared import SharedBuffer, SharedLeague
from trainer.inference import InferenceServer
from trainer.worker import worker_fn
from config import VexConfig
import torch.multiprocessing as mp
from model.model import AgentMLP
import time 
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    config = VexConfig()
    base_model = AgentMLP(config).to('cpu')
    # Create manager for all shared synchronization primitives
    with mp.Manager() as manager:
        buffer = SharedBuffer(config, manager)
        league = SharedLeague(config, manager)
        inference_server = InferenceServer(buffer, league, config, 'cuda')
        league.update_learner_param(base_model) # initialize league with the base model.
        # inference server process
        server_proc = mp.Process(target=inference_server.run, args=())
        server_proc.start()
        
        workers_proc = []
        for worker_id in range(config.n_workers):
            p = mp.Process(target=worker_fn, args=(worker_id, buffer, league, config))
            p.start()
            workers_proc.append(p)
        
        # Keep manager alive while processes are running
        server_proc.join()
        for p in workers_proc:
            p.join()
        
    # update_interval = 2.0  # We want to update the model every 2 seconds
    # start_time = time.perf_counter()

    # while True:
    #     # 1. Wait for the update interval (Simulating training time)
    #     time.sleep(update_interval)
        
    #     # 2. Check the "Pile"
    #     chunks_waiting = buffer.ready_indices.qsize()
    #     steps_waiting = chunks_waiting * config.chunk_size
        
    #     # 3. PULL EVERYTHING
    #     # This matches the worker production exactly
    #     actual_batch = buffer.pull_from_buffer(steps_waiting)
        
    #     # 4. Reporting
    #     elapsed = time.perf_counter() - start_time
    #     print(f"--- Interval: {update_interval}s ---")
    #     print(f"Required Batch Size to keep up: {steps_waiting}")
    #     print(f"Current System Production FPS: {steps_waiting / update_interval:.2f}")
        
    #     start_time = time.perf_counter()
