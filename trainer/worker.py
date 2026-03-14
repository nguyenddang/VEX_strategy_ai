import torch 
import torch.multiprocessing as mp
from trainer.shared import SharedBuffer, SharedLeague
from config import VexConfig
from env.env import VexEnv

from typing import Dict
import time 
from model.model import GeniusFormer

        
def zeros_buffer(buffer: Dict[str, torch.Tensor]):
    for key in buffer:
        buffer[key].zero_()

def worker_decentralized_fn(
    worker_id, 
    buffer: SharedBuffer,
    league: SharedLeague,
    config: VexConfig,
):
    # decentralized: worker has its own copy of learner and opponent. 
    torch.set_num_threads(1)
    time.sleep(worker_id * 0.01) 
    env = VexEnv(config)
    opponent_model = GeniusFormer(config)
    learner_model = GeniusFormer(config)
    learner_model.eval()
    opponent_model.eval() 
    worker_learner_version = league.learner_version.value
    local_buffer = {
        'core_obs': torch.zeros((2, config.max_actions, config.core_obs_dim), dtype=torch.float32),
        'ball_obs': torch.zeros((2, config.max_actions, config.n_balls, config.ball_obs_dim), dtype=torch.float32),
        'legal_masks': torch.zeros((2, config.max_actions, config.n_primary_actions), dtype=torch.bool),
        'rewards': torch.zeros((2, config.max_actions), dtype=torch.float32),
        'actions': torch.zeros((2, config.max_actions, 4), dtype=torch.long),
        'values': torch.zeros((2, config.max_actions + 1), dtype=torch.float32),
        'move_masks': torch.zeros((2, config.max_actions), dtype=torch.bool),
        'log_probs': torch.zeros((2, config.max_actions), dtype=torch.float32),
        'learner_versions': torch.zeros((config.max_actions,), dtype=torch.float32),
    }
    print(f"Worker {worker_id} started.", flush=True)
    while True:
        opp_idx, p, n, param = league.sample_opponent(worker_id)
        torch.nn.utils.vector_to_parameters(param, opponent_model.parameters())
        zeros_buffer(local_buffer)

        # check if learner has been updated. If yes, pull 
        with league.learner_lock:
            if league.learner_version.value != worker_learner_version:
                torch.nn.utils.vector_to_parameters(league.learner_param, learner_model.parameters())
                worker_learner_version = league.learner_version.value
        
        for model in [learner_model, opponent_model]:
            model.reset_kv_cache()

        env_out = env.reset()
        done, legal_actions, observations, rewards, timestep = \
            env_out['done'], env_out['legal_actions'], env_out['observations'], env_out['rewards'], env_out['timestep']
        
        while not done:
            for p_idx, robot_key in enumerate(['robot_red', 'robot_blue']):
                local_buffer['core_obs'][p_idx, timestep].copy_(observations[robot_key]['core_obs'])
                local_buffer['ball_obs'][p_idx, timestep].copy_(observations[robot_key]['ball_obs'])
                local_buffer['legal_masks'][p_idx, timestep].copy_(legal_actions[robot_key])
            with torch.no_grad():
                # learner inference
                learner_out = learner_model(
                    local_buffer['core_obs'][:, [timestep]], # (2, some size of block, core_obs_dim)
                    local_buffer['ball_obs'][:, [timestep]], # (2, some size of block, n_balls, ball_obs_dim)
                    local_buffer['legal_masks'][:, timestep], # (2, some size of block, n_primary_actions)
                    do_inference=True)
                opponent_out = opponent_model(
                    local_buffer['core_obs'][1, [timestep]].unsqueeze(0), # (1, some size of block, core_obs_dim)
                    local_buffer['ball_obs'][1, [timestep]].unsqueeze(0), # (1, some size of block, n_balls, ball_obs_dim)
                    local_buffer['legal_masks'][1, timestep].unsqueeze(0), # (1, some size of block, n_primary_actions)
                    do_inference=True
                )
            # copy outputs to local buffer 
            local_buffer['actions'][0, timestep] = learner_out['actions'][0]
            local_buffer['actions'][1, timestep] = opponent_out['actions'][0]
            local_buffer['values'][:, timestep] = learner_out['values']
            local_buffer['move_masks'][0, timestep] = learner_out['move_mask'][0]
            local_buffer['move_masks'][1, timestep] = opponent_out['move_mask'][0]
            local_buffer['log_probs'][0, timestep] = learner_out['log_prob'][0]
            local_buffer['log_probs'][1, timestep] = opponent_out['log_prob'][0]
            local_buffer['learner_versions'][timestep] = worker_learner_version
            act = {
                'robot_red': learner_out['actions'][0].tolist(),
                'robot_blue': opponent_out['actions'][0].tolist(),
            }
            env_out = env.step(act)
            # save reward
            local_buffer['rewards'][0, timestep] = env_out['rewards']['robot_red']
            local_buffer['rewards'][1, timestep] = env_out['rewards']['robot_blue']

            done, legal_actions, observations, rewards, timestep = \
                env_out['done'], env_out['legal_actions'], env_out['observations'], env_out['rewards'], env_out['timestep']

        
        delta = 0.01/(n * p)
        league.update_quality(opp_idx, delta)
        buffer.push_to_buffer(local_buffer)