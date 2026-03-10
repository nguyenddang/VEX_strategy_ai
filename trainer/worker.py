import torch 
import torch.multiprocessing as mp
from trainer.shared import SharedBuffer, SharedLeague
from config import VexConfig
from env.env import VexEnv

from typing import Dict

        
def zeros_buffer(buffer: Dict[str, torch.Tensor]):
    for key in buffer:
        buffer[key].zero_()


@torch.no_grad()
def worker_fn(
    worker_id, 
    buffer: SharedBuffer,
    league: SharedLeague,
    config: VexConfig,
):
    torch.set_num_threads(1)
    env = VexEnv(config)
    print(f"Worker {worker_id} started.")
    local_buffer = {
            'core_obs': torch.zeros((2, config.chunk_size, config.core_obs_dim), dtype=torch.float32),
            'ball_obs': torch.zeros((2, config.chunk_size, config.n_balls, config.ball_obs_dim), dtype=torch.float32),
            'legal_masks': torch.zeros((2, config.chunk_size, config.n_primary_actions), dtype=torch.bool),
            'actions': torch.zeros((2, config.chunk_size, 4), dtype=torch.long),
            'rewards': torch.zeros((2, config.chunk_size), dtype=torch.float32),
            'values': torch.zeros((2, config.chunk_size + 1), dtype=torch.float32), # +1 for bootstrapping value of last state.
            'move_masks': torch.zeros((2, config.chunk_size), dtype=torch.bool),
        }
    while True:
        zeros_buffer(local_buffer)
        opp_idx, param_version, p, q, n_snapshots = league.sample_opponent(worker_id)
        print(f"Worker {worker_id} playing {opp_idx} version {param_version}")
        env_out = env.reset()
        done, legal_actions, observations, rewards, timestep = \
            env_out['done'], env_out['legal_actions'], env_out['observations'], env_out['reward'], env_out['timestep']
        while not done:
            idx = timestep % config.chunk_size
            # copy to slots for inference and update part of local buffer.
            for p_idx, robot_key in enumerate(['robot_red', 'robot_blue']):
                buffer.temp['core_obs'][worker_id, p_idx].copy_(observations[robot_key]['core_obs'])
                buffer.temp['ball_obs'][worker_id, p_idx].copy_(observations[robot_key]['ball_obs'])
                buffer.temp['legal_masks'][worker_id, p_idx].copy_(legal_actions[robot_key])
                local_buffer['core_obs'][p_idx, idx].copy_(observations[robot_key]['core_obs'])
                local_buffer['ball_obs'][p_idx, idx].copy_(observations[robot_key]['ball_obs'])
                local_buffer['legal_masks'][p_idx, idx].copy_(legal_actions[robot_key])
            # inference 
            buffer.inference_queue.put((worker_id, opp_idx, param_version)) # request to inference gpu. 
            buffer.res_events[worker_id].wait()
            buffer.res_events[worker_id].clear() 
            

            actions = {
                'robot_red': buffer.temp['actions'][worker_id, 0].tolist(),
                'robot_blue': buffer.temp['actions'][worker_id, 1].tolist(),
            }
            env_out = env.step(actions)
            
            # collect results and update local_buffer
            for p_idx, robot_key in enumerate(['robot_red', 'robot_blue']):
                local_buffer['actions'][p_idx, idx].copy_(buffer.temp['actions'][worker_id, p_idx])
                local_buffer['rewards'][p_idx, idx] = env_out['reward'][robot_key] # use reward for transition s-> s'
                local_buffer['values'][p_idx, idx].copy_(buffer.temp['values'][worker_id, p_idx])
                local_buffer['move_masks'][p_idx, idx].copy_(buffer.temp['move_masks'][worker_id, p_idx])
                            
            if (idx + 1) == config.chunk_size and timestep > 0:
                if not env_out['done']: # if done, do not booststrap and just use 0 as the value of last state.
                    # get bootstrapping value for the last state.
                    for p_idx, robot_key in enumerate(['robot_red', 'robot_blue']):
                        buffer.temp['core_obs'][worker_id, p_idx].copy_(env_out['observations'][robot_key]['core_obs'])
                        buffer.temp['ball_obs'][worker_id, p_idx].copy_(env_out['observations'][robot_key]['ball_obs'])
                        buffer.temp['legal_masks'][worker_id, p_idx].copy_(env_out['legal_actions'][robot_key])
                    buffer.inference_queue.put((worker_id, opp_idx, param_version)) # request to inference gpu. 
                    buffer.res_events[worker_id].wait()
                    buffer.res_events[worker_id].clear() 
                    # copy bootstrapping value to local buffer.
                    for p_idx in range(2):
                        local_buffer['values'][p_idx, -1].copy_(buffer.temp['values'][worker_id, p_idx])
                # reset local buffer and push to shared buffer.
                buffer.push_to_buffer(local_buffer)
                zeros_buffer(local_buffer)
            
            done, legal_actions, observations, rewards, timestep = \
                env_out['done'], env_out['legal_actions'], env_out['observations'], env_out['reward'], env_out['timestep']
            
            if done: 
                # update new quality to buffer. 
                # refer to Eq 14 Appendix N of Dota 2 paper.
                delta = 0.01 / (p * n_snapshots)
                if env_out['score']['robot_red'] > env_out['score']['robot_blue']:
                    new_q = q - delta
                    league.update_quality(opp_idx, param_version, new_q) # learner won, decrease opponent quality.
                