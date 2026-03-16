import os 

import trueskill
from trueskill import rate_1vs1

from config import VexConfig
from evaluator.eval_worker import eval_workers
from model.mlp import MLP

import torch 
import torch.multiprocessing as mp

import time 
from tqdm import tqdm
import random 

class Evaluator:
    
    def __init__(self, checkpoint_path: str, n_games_per_matchup: int = 50, n_workers: int = 10):
        ckpts = [f for f in os.listdir(checkpoint_path) if f.startswith("learner") and f.endswith(".pt")]
        ckpts = sorted(ckpts, key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)))
        ckpts = ckpts[1::2] if len(ckpts) > 10 else ckpts
        ckpts = ['random'] + ckpts # additional random for baseline 
        
        self.matchups = mp.Queue()
        temp = []
        for i in range(len(ckpts)):
            for j in range(i+1, len(ckpts)):
                for _ in range(n_games_per_matchup):
                    temp.append((i, j))
        random.shuffle(temp)
        for pair in temp:
            self.matchups.put(pair)
            
        print(f"Total matchups: {self.matchups.qsize()}. {n_games_per_matchup} games per matchup, total {self.matchups.qsize()*n_games_per_matchup} games.")
        temp_ckpt = torch.load(os.path.join(checkpoint_path, ckpts[-1]), map_location='cpu')
        temp_config = VexConfig()
        for key, value in temp_ckpt['config'].items():
            setattr(temp_config, key, value)
        temp_model = MLP(temp_config)
        total_params = sum(p.numel() for p in temp_model.parameters())
        self.param_bank = torch.zeros((len(ckpts), total_params), dtype=torch.float32).share_memory_()
        # load checkpoint parameters into param_bank
        for idx, ckpt in enumerate(ckpts):
            if ckpt != 'random':
                ckpt_data = torch.load(os.path.join(checkpoint_path, ckpt), map_location='cpu')
                temp_model.load_state_dict(ckpt_data['model'])
            else:
                temp_model.apply(lambda m: torch.nn.init.normal_(m.weight, mean=0.0, std=0.02) if isinstance(m, torch.nn.Linear) else None)
            self.param_bank[idx].copy_(torch.nn.utils.parameters_to_vector(temp_model.parameters()))
        
        self.n_workers = n_workers
        self.n_games_per_matchup = n_games_per_matchup
        # trueskill cache 
        self.mus = torch.full((len(ckpts),), 25.0, dtype=torch.float32).share_memory_()
        self.sigmas = torch.full((len(ckpts),), 25.0/3, dtype=torch.float32).share_memory_()
        self.ts_env = trueskill.TrueSkill(mu=25.0, sigma=25.0/3, beta=25.0/6, tau=0.0, draw_probability=0.02)
        self.config = temp_config
        self.ckpts = ckpts
        self.lock = mp.Lock()
        self.progress = mp.Value('i', 0)  # shared int
    def get_next_matchup(self):
        try:
            idx1, idx2 = self.matchups.get(timeout=1)
        except:
            return None
        param1 = self.param_bank[idx1]
        param2 = self.param_bank[idx2]
        return idx1, idx2, param1, param2
    
    def update_trueskill(self, idx1: int, idx2: int, idx1_won:bool, idx2_won:bool):
        with self.progress.get_lock():
            self.progress.value += 1
        with self.lock:
            rating1 = self.ts_env.create_rating(mu=self.mus[idx1].item(), sigma=self.sigmas[idx1].item())
            rating2 = self.ts_env.create_rating(mu=self.mus[idx2].item(), sigma=self.sigmas[idx2].item())
            if idx1_won:
                new_rating1, new_rating2 = rate_1vs1(rating1, rating2)
            elif idx2_won:
                new_rating2, new_rating1 = rate_1vs1(rating2, rating1)
            else:
                return
            self.mus[idx1] = new_rating1.mu
            self.sigmas[idx1] = new_rating1.sigma
            self.mus[idx2] = new_rating2.mu
            self.sigmas[idx2] = new_rating2.sigma
            
    def run(self):
        # run evaluation. 
        # create processes 
        total_games = self.matchups.qsize()
        processes = []
        for worker_id in range(self.n_workers):
            p = mp.Process(target=eval_workers, args=(worker_id, self, self.config))
            p.start()
            processes.append(p)
            
        with tqdm(total=total_games, desc="Evaluating", unit="game") as pbar:
            last = 0
            while last < total_games:
                time.sleep(0.2)
                with self.progress.get_lock():
                    current = self.progress.value
                pbar.update(current - last)
                last = current
        # wait for processes to finish and terminate
        for p in processes:
            p.join()
        
        print("All evaluation matchups completed.")
        # print final trueskill ratings
        for idx in range(len(self.mus)):
            print(f"Model {self.ckpts[idx]}: mu={self.mus[idx].item():.2f}, sigma={self.sigmas[idx].item():.2f}")

