import torch 
import os 
import trueskill
from config import VexConfig
from model.mlp import MLP

import torch.multiprocessing as mp
class Evaluator:
    
    def __init__(self, checkpoint_path: str, n_games_per_matchup: int = 50):
        ckpts = [f for f in os.listdir(checkpoint_path) if f.startswith("learner") and f.endswith(".pt")]
        ckpts = sorted(ckpts, key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)))
        ckpts = ckpts[1::2]
        self.matchups = {}
        for i in range(len(ckpts)):
            for j in range(i+1, len(ckpts)):
                self.matchups[(i,j)] = 0
        print(f"Total matchups: {len(self.matchups)}. {n_games_per_matchup} games per matchup, total {len(self.matchups)*n_games_per_matchup} games.")
        temp_ckpt = torch.load(os.path.join(checkpoint_path, ckpts[0]), map_location='cpu')
        temp_config = VexConfig()
        for key, value in temp_ckpt['config'].items():
            setattr(temp_config, key, value)
        temp_model = MLP(temp_config)
        total_params = sum(p.numel() for p in temp_model.parameters())
        self.param_bank = torch.zeros((len(ckpts), total_params), dtype=torch.float32).share_memory_()
        # load checkpoint parameters into param_bank
        for idx, ckpt in enumerate(ckpts):
            ckpt_data = torch.load(os.path.join(checkpoint_path, ckpt), map_location='cpu')
            temp_model.load_state_dict(ckpt_data['model'])
            self.param_bank[idx].copy_(torch.nn.utils.parameters_to_vector(temp_model.parameters()))
        
        self.n_games_per_matchup = n_games_per_matchup
        # trueskill cache 
        self.mus = torch.zeros((len(ckpts),), dtype=torch.float32)
        self.sigmas = torch.ones((len(ckpts),), dtype=torch.float32) * (25/3)
        self.ts_env = trueskill.TrueSkill(mu=25.0, sigma=25.0/3, beta=25.0/6, tau=0.0, draw_probability=0.02)
        
        self.update_lock = mp.Lock()
    def get_next_matchup(self):
        # randomly sample a pair that has not reached n_games_per_matchup yet.
        valid_pairs = [pair for pair, count in self.matchups.items() if count < self.n_games_per_matchup]
        if not valid_pairs:
            return None
        pair = valid_pairs[torch.randint(len(valid_pairs), (1,)).item()]
        self.matchups[pair] += 1
        return pair[0], pair[1], self.param_bank[pair[0]], self.param_bank[pair[1]]
    
    def update_trueskill(self, idx1: int, idx2: int, )
# checkpoint_path = 'checkpoints_0'
# # get all .pt file starting with "learner" in checkpoint_path and sort by modified time.
# checkpoints = [f for f in os.listdir(checkpoint_path) if f.startswith("learner") and f.endswith(".pt")]
# checkpoints = sorted(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)))
# # get every 2 checkpoints to evaluate.
# checkpoints = checkpoints[1::2]

# # form pairwise matchups between checkpoints. (round robin style)
# n_games_per_matchup = 16
# matchups = []
# for i in range(len(checkpoints)):
#     for j in range(i+1, len(checkpoints)):
#         matchups.append((checkpoints[i], checkpoints[j]))
# print(f"Total matchups: {len(matchups)}")
# print(f"matchups: {matchups}")

