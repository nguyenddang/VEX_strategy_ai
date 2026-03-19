import os 

import trueskill
from trueskill import rate_1vs1

from config import VexConfig
from model.mlp import MLP

from trainer.shared import SharedLeague
import torch 
import torch.multiprocessing as mp
import torch.nn as nn

class TrainEvaluator:
    
    def __init__(self, config: VexConfig, league: SharedLeague):
        self.evaluator_capacity = config.evaluator_capacity
        self.n_games_per_pair = config.n_games_per_pair 
        local_model = MLP(config)
        total_params = sum(p.numel() for p in local_model.parameters())
        
        self.ref_params = torch.zeros((self.evaluator_capacity, total_params), dtype=torch.float32).share_memory_()
        self.ref_mus = torch.full((self.evaluator_capacity,), float('-inf'), dtype=torch.float32).share_memory_()
        self.ref_sigmas = torch.full((self.evaluator_capacity,), 25.0/3, dtype=torch.float32).share_memory_()
        self.lastest_ref_idx = mp.Value('i', -1)
        
        # pre_init random model. 
        rand_param = torch.nn.utils.parameters_to_vector(local_model.parameters()).detach().cpu()
        self.ref_params[0].copy_(rand_param)
        self.ref_mus[0] = 0.0
        self.ref_sigmas[0] = 1.0
        self.lastest_ref_idx.value = 0
        
        self.learner_param_eval = torch.zeros((total_params,), dtype=torch.float32).share_memory_()
        self.learner_version_eval = mp.Value('i', -1)
        
        self.test_mu = mp.Value('d', 0.0)
        self.test_sigma = mp.Value('d', 25.0/3)
        self.test_n = mp.Value('i', 0) # games played 
        self.test_version = mp.Value('i', -1) 
        self.test_param = torch.zeros((total_params,), dtype=torch.float32).share_memory_()        
        self.lock = mp.Lock()
        
        trueskill.setup(mu=0.0, sigma=25.0/3, beta=25.0/6, tau=0, draw_probability=0.02)
        
    
    def update_learner_param(self, model: nn.Module, version: int):
        # called by trainer gpu
        param = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu()
        with self.lock:
            self.learner_param_eval.copy_(param)
            self.learner_version_eval.value = version
            if self.test_version.value == -1:
                # first time update, directly put in test slot. 
                self.test_param.copy_(param)
                self.test_version.value = version
            
    def get_next_matchup(self):
        with self.lock:
            if self.test_version.value == -1:
                return None
            # only play against ref against that is 10 trueskill away 
            mus = self.ref_mus[:self.lastest_ref_idx.value+1]
            valid_idxs = (self.test_mu.value - mus).abs() < 10.0
            if valid_idxs.sum() == 0:
                # fall back to play against latest ref if no valid matchup.
                valid_idxs[self.lastest_ref_idx.value] = True
            # pick randomly from valid matchups
            valid_idxs = torch.where(valid_idxs)[0]
            ref_idx = valid_idxs[torch.randint(len(valid_idxs), (1,)).item()].item()
            return self.test_version.value, ref_idx

    def update_trueskill(self, test_version: int, ref_idx: int, test_won: bool, ref_won: bool):
        with self.lock:
            if test_version != self.test_version.value:
                # worker is behind, ignore result. 
                return
            if self.test_n.value + 1 <= self.n_games_per_pair or self.test_version.value == self.learner_version_eval.value:
                # if game played not reach or new learner is not updated, update trueskill.  
                test_rating = trueskill.Rating(self.test_mu.value, self.test_sigma.value)
                ref_rating = trueskill.Rating(self.ref_mus[ref_idx].item(), self.ref_sigmas[ref_idx].item())
                if test_won:
                    new_test_rating, _ = rate_1vs1(test_rating, ref_rating)
                elif ref_won:
                    _, new_test_rating = rate_1vs1(ref_rating, test_rating)
                else:
                    new_test_rating, _ = rate_1vs1(test_rating, ref_rating, drawn=True)
                self.test_mu.value = new_test_rating.mu
                self.test_sigma.value = new_test_rating.sigma
                self.test_n.value += 1
                return 
            if self.test_n.value + 1 > self.n_games_per_pair and self.test_version.value != self.learner_version_eval.value:
                # if games reached and new learner is here, graduate test to ref. 
                last_ref_mu = self.ref_mus[self.lastest_ref_idx.value].item()
                if self.test_mu.value - last_ref_mu > 5.0:
                    replace_idx = torch.argmin(self.ref_mus).item()
                    print(f"[EVALUATOR] Graduating version {self.test_version.value} to ref slot {replace_idx}.", flush=True)
                    self.ref_params[replace_idx].copy_(self.test_param)
                    self.ref_mus[replace_idx] = self.test_mu.value
                    self.ref_sigmas[replace_idx] = self.test_sigma.value
                    self.lastest_ref_idx.value = replace_idx
                    
                # update test to new learner
                self.test_param.copy_(self.learner_param_eval)
                self.test_version.value = self.learner_version_eval.value
                self.test_mu.value = self.test_mu.value # keep the same initial skill for new learner
                self.test_sigma.value = 25.0/3
                self.test_n.value = 0
                return 
            
    def state_dict(self):
        with self.lock:
            return {
                "ref_params": self.ref_params.clone(),
                "ref_mus": self.ref_mus.clone(),
                "ref_sigmas": self.ref_sigmas.clone(),
                "lastest_ref_idx": self.lastest_ref_idx.value,
                "learner_param_eval": self.learner_param_eval.clone(),
                "learner_version_eval": self.learner_version_eval.value,
                "test_param": self.test_param.clone(),
                "test_version": self.test_version.value,
                "test_mu": self.test_mu.value,
                "test_sigma": self.test_sigma.value,
                "test_n": self.test_n.value,
            }
            
    def load_state_dict(self, state_dict):
        with self.lock:
            self.ref_params.copy_(state_dict["ref_params"])
            self.ref_mus.copy_(state_dict["ref_mus"])
            self.ref_sigmas.copy_(state_dict["ref_sigmas"])
            self.lastest_ref_idx.value = state_dict["lastest_ref_idx"]
            self.learner_param_eval.copy_(state_dict["learner_param_eval"])
            self.learner_version_eval.value = state_dict["learner_version_eval"]
            self.test_param.copy_(state_dict["test_param"])
            self.test_version.value = state_dict["test_version"]
            self.test_mu.value = state_dict["test_mu"]
            self.test_sigma.value = state_dict["test_sigma"]
            self.test_n.value = state_dict["test_n"]
                
                
        
            
    
        
        
    
        
        