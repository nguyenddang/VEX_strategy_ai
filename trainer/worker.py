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
        'core_obs': torch.zeros((config.max_actions, config.core_obs_dim), dtype=torch.float32),
        'ball_obs': torch.zeros((config.max_actions, config.n_balls, config.ball_obs_dim), dtype=torch.float32),
        'legal_masks': torch.zeros((config.max_actions, config.n_primary_actions), dtype=torch.bool),
        'rewards': torch.zeros((config.max_actions), dtype=torch.float32),
        'actions': torch.zeros((config.max_actions, 4), dtype=torch.long),
        'values': torch.zeros((config.max_actions + 1), dtype=torch.float32),
        'move_masks': torch.zeros((config.max_actions,), dtype=torch.bool),
        'log_probs': torch.zeros((config.max_actions,), dtype=torch.float32),
        'learner_versions': torch.zeros((1,), dtype=torch.float32),
        'red_score': torch.zeros((1,), dtype=torch.float32),
        'blue_score': torch.zeros((1,), dtype=torch.float32),
    }
    print(f"Worker {worker_id} started.", flush=True)
    learner_key, opp_key = 'robot_red', 'robot_blue'
    while True:
        opp_idx, p, n, param = league.sample_opponent(worker_id)
        # load opponent and learner parameters
        torch.nn.utils.vector_to_parameters(param, opponent_model.parameters())
        with league.learner_lock:
            if league.learner_version.value != worker_learner_version:
                torch.nn.utils.vector_to_parameters(league.learner_param, learner_model.parameters())
                worker_learner_version = league.learner_version.value
        # reset kv caches 
        for m in [opponent_model, learner_model]:
            m.reset_kv_cache()
        zeros_buffer(local_buffer)
        local_buffer['learner_versions'].fill_(worker_learner_version)

        env_out = env.reset()
        done, legal_actions, observations, rewards, timestep = \
            env_out['done'], env_out['legal_actions'], env_out['observations'], env_out['rewards'], env_out['timestep']
        
        while not done:
            local_buffer['core_obs'][timestep].copy_(observations[learner_key]['core_obs'])
            local_buffer['ball_obs'][timestep].copy_(observations[learner_key]['ball_obs'])
            local_buffer['legal_masks'][timestep].copy_(legal_actions[learner_key])
            
            with torch.no_grad():
                learner_out = learner_model(
                    observations[learner_key]['core_obs'].view(1, 1, -1), # (1, 1, core_obs_dim)
                    observations[learner_key]['ball_obs'].view(1, 1, config.n_balls, -1), # (1, 1, n_balls, ball_obs_dim)
                    legal_actions[learner_key].view(1, -1), # (1, n_primary_actions)
                    do_inference=True
                )
                opponent_out = opponent_model(
                    observations[opp_key]['core_obs'].view(1, 1, -1), # (1, 1, core_obs_dim)
                    observations[opp_key]['ball_obs'].view(1, 1, config.n_balls, -1), # (1, 1, n_balls, ball_obs_dim)
                    legal_actions[opp_key].view(1, -1), # (1, n_primary_actions)
                    do_inference=True
                )
            # copy outputs to local buffer 
            local_buffer['actions'][timestep].copy_(learner_out['actions'][0])
            local_buffer['values'][timestep].copy_(learner_out['values'][0])
            local_buffer['move_masks'][timestep].copy_(learner_out['move_mask'][0])
            local_buffer['log_probs'][timestep].copy_(learner_out['log_prob'][0])
            
            act = {
                'robot_red': learner_out['actions'][0].tolist(),
                'robot_blue': opponent_out['actions'][0].tolist(),
            }
            env_out = env.step(act)
            # save reward
            local_buffer['rewards'][timestep] = env_out['rewards']['robot_red']

            done, legal_actions, observations, rewards, timestep = \
                env_out['done'], env_out['legal_actions'], env_out['observations'], env_out['rewards'], env_out['timestep']

        local_buffer['red_score'][0] = env_out['score']['robot_red']
        local_buffer['blue_score'][0] = env_out['score']['robot_blue']

        delta = 0.01/(n * p)
        red_won = env_out['score']['robot_red'] > env_out['score']['robot_blue']
        if red_won:
            league.update_quality(opp_idx, delta)
        buffer.push_to_buffer(local_buffer)