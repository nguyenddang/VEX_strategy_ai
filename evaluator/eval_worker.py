from config import VexConfig
from model.mlp import MLP
from env.env import VexEnv

import torch 
import time
def eval_workers(
    worker_id: int,
    evaluator,
    config: VexConfig,
):
    time.sleep(worker_id * 0.01)
    torch.set_num_threads(1)
    env = VexEnv(config)
    model_1 = MLP(config)
    model_2 = MLP(config)
    model_1.eval()
    model_2.eval()
    print(f"Eval Worker {worker_id} started.", flush=True)
    while True:
        pair = evaluator.get_next_matchup()
        if pair is None:
            # shutdown signal, all matchups are done.
            print(f"Eval Worker {worker_id} received shutdown signal. All matchups are done. Exiting.", flush=True)
            break
        idx1, idx2, param1, param2 = pair    
        torch.nn.utils.vector_to_parameters(param1, model_1.parameters())
        torch.nn.utils.vector_to_parameters(param2, model_2.parameters())
        env_out = env.reset()
        done, legal_actions, observations, rewards, timestep = \
            env_out['done'], env_out['legal_actions'], env_out['observations'], env_out['rewards'], env_out['timestep']
        while not done:
            with torch.no_grad():
                out1 = model_1(
                    observations['robot_red']['core_obs'].view(1, -1), # (1, core_obs_dim)
                    observations['robot_red']['ball_obs'].view(1, config.n_balls, -1), # (1, n_balls, ball_obs_dim)
                    legal_actions['robot_red'].view(1, -1), # (1, n_primary_actions)
                    inference=True
                )
                out2 = model_2(
                    observations['robot_blue']['core_obs'].view(1, -1), # (1, core_obs_dim)
                    observations['robot_blue']['ball_obs'].view(1, config.n_balls, -1), # (1, n_balls, ball_obs_dim)
                    legal_actions['robot_blue'].view(1, -1), # (1, n_primary_actions)
                    inference=True
                )
            actions = {
                'robot_red': out1['actions'][0].tolist(), # [discrete, x, y, theta]
                'robot_blue': out2['actions'][0].tolist(),
            }
            env_out = env.step(actions)
            done, legal_actions, observations, rewards, timestep = \
                env_out['done'], env_out['legal_actions'], env_out['observations'], env_out['rewards'], env_out['timestep']
        
        red_won = env_out['score']['robot_red'] > env_out['score']['robot_blue']
        blue_won = env_out['score']['robot_blue'] > env_out['score']['robot_red']   
        evaluator.update_trueskill(idx1, idx2, red_won, blue_won)
    