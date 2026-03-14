# from model.model import GeniusFormer
# from config import VexConfig
# import torch
# import time
# import torch.multiprocessing as mp
# from trainer.shared import SharedBuffer, SharedLeague
# from trainer.worker import worker_decentralized_fn

# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     config = VexConfig()
#     config.n_workers = 1
#     config.buffer_capacity = 16
#     config.train_episodes = 8

#     shared_buffer = SharedBuffer(config)
#     shared_league = SharedLeague(config)
#     learner = GeniusFormer(config)

#     shared_league.update_learner_param(learner)
#     shared_league.update_latest_snapshot(learner)

#     processes = []
#     for worker_id in range(config.n_workers):
#         p = mp.Process(target=worker_decentralized_fn, args=(worker_id, shared_buffer, shared_league, config))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

import torch
name = torch.cuda.get_device_name(0)
print(f"Using device: {name.split(' ')[-1] == 'H200'}")