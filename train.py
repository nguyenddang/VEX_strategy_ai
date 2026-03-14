from config import VexConfig
import torch.multiprocessing as mp
from trainer.trainer import Trainer
import torch 

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    config = VexConfig()
    config.train_device = "cuda:0"
    config.n_workers = 90
    config.buffer_capacity = 256
    config.train_episodes = 32
    config.log_wandb = False
    # config.mini_train_episodes = 32 if torch.cuda.get_device_name(0).split(' ')[-1] == 'H200' else 16
    config.mini_train_episodes = 32

    trainer = Trainer(config)
    trainer.train()