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
    config.n_workers = 88
    config.buffer_capacity = 200
    config.train_episodes = 16
    config.n_embd = 256
    config.n_save_learner_ckpts = 250
    config.n_save_all_ckpts = 25
    config.log_wandb = True
    config.mini_train_episodes = 16 
    config.steps_per_iteration = 32
    config.update_league = 10
    config.save_ckpt_path = "checkpoints_7"
    config.compile = True 
    config.load_ckpt_path = 'checkpoints_6/all.pt'
    config.resume_training = True 
    config.n_eval_workers = 5
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    os.makedirs(config.save_ckpt_path, exist_ok=True)
    trainer = Trainer(config)
    trainer.train()