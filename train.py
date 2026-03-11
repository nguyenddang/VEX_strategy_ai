from config import VexConfig
import torch.multiprocessing as mp
from trainer.trainer import Trainer
if __name__ == "__main__":
    mp.set_start_method("spawn")
    config = VexConfig()
    config.train_device = "cuda:0"
    config.inference_server_device = "cuda:1"
    config.n_workers = 86
    config.steps_per_iteration = 16
    config.update_league = 10
    config.train_batch_size = 8192 * 2
    config.chunk_size = 32

    trainer = Trainer(config)
    trainer.train()