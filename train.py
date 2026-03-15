from config import VexConfig
import torch.multiprocessing as mp
from trainer.trainer import Trainer
import torch 
import os 
if __name__ == "__main__":
    torch.manual_seed(42)
    mp.set_start_method("spawn", force=True)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    config = VexConfig()
    config.train_device = "cuda:0"
    config.n_workers = 90
    config.buffer_capacity = 150
    config.train_episodes = 32
    config.n_embd = 128
    config.n_save_learner_ckpts = 25
    config.n_save_all_ckpts = 25
    config.log_wandb = False
    config.mini_train_episodes = 32 if torch.cuda.get_device_name(0).split(' ')[-1] == 'H200' else 16
    config.steps_per_iteration = 16
    config.save_ckpt_path = "checkpoints_0"
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    os.makedirs(config.save_ckpt_path, exist_ok=True)
    trainer = Trainer(config)
    trainer.train()